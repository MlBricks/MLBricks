
#!/usr/bin/env python3
"""MLBricks Learned ESA vs Learned Long-Memory ESA.

Two workers, one model per GPU. Synthetic associative recall, not natural LM.
Full-state atomic resume. Identical shared initial linear/embedding weights and
identical deterministic training examples at every optimizer step.

Requires PyTorch and NumPy. CPU smoke-test mode is supported.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import random
import time
import traceback
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KEY_MARK, IS_MARK, END_MARK, QUERY_MARK, ANSWER_MARK = 2, 3, 4, 5, 6
KEY_BASE, N_KEYS = 16, 32
VALUE_BASE, N_VALUES = 64, 32
FILLER_LOW, FILLER_HIGH = 128, 256
DEFAULT_TIMESCALES = [16, 64, 256, 1024, 4096, 8192]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


def chunk_scan(a, v, initial=None, chunk_size=32):
    """Exact-in-real-arithmetic associative affine recurrence, chunked FP32 scan.

    E_t = a_t * E_(t-1) + v_t. a and v shaped [B,T,H,Dh].
    Zero-length chunks not allowed. Preserve recurrent state across chunks.
    """
    B, T, H, Dh = a.shape
    state = (v.new_zeros(B, H, Dh) if initial is None else initial).float()
    result = []
    for i in range(0, T, chunk_size):
        aa = a[:, i:i+chunk_size].float()
        vv = v[:, i:i+chunk_size].float()
        # A >= ~0.9 initially, so product of 32 terms stays well in FP32 range.
        pref = torch.cumprod(aa, dim=1)
        val = pref * (state[:, None] + torch.cumsum(vv / pref, dim=1))
        state = val[:, -1]
        result.append(val)
    return torch.cat(result, dim=1), state




class ESAFullMixer(nn.Module):
    """Compare learned-bounds ESA with a bounded-write long-memory extension.

    Both use precisely the same Q/G/V, retention-coupled writing and gated readout.
    In the long_memory variant, each head learns a retention timescale multiplier.
        A0 = alpha_h + (beta_h-alpha_h) sigmoid(G)
        stretch_h = exp(4 tanh(raw_stretch_h/4))
        A_lm = A0 ** (1/stretch_h)
    raw_stretch starts at zero => A_lm == A0 to floating-point tolerance.
    The timescale multiplier is limited smoothly to [exp(-4), exp(4)].
    Write U=(1-A)*tanh(V), which keeps state bounded for all contexts.
    """
    def __init__(self,dim,heads,variant):
        super().__init__()
        assert variant in ('learned','learned_lm')
        assert dim % heads == 0
        self.dim,self.heads,self.dh,self.variant=dim,heads,dim//heads,variant
        self.qgv=nn.Linear(dim,3*dim,bias=False)
        self.W_O=nn.Linear(dim,dim,bias=False)
        hard_max=0.9999
        amin=0.80/hard_max
        bmin=(0.995-0.80)/(hard_max-0.80)
        self.raw_alpha=nn.Parameter(torch.full((heads,),math.log(amin/(1-amin))))
        self.raw_beta=nn.Parameter(torch.full((heads,),math.log(bmin/(1-bmin))))
        if variant=='learned_lm':
            self.raw_stretch=nn.Parameter(torch.zeros(heads))

    def bounds(self):
        alpha=0.9999*torch.sigmoid(self.raw_alpha)
        beta=alpha+(0.9999-alpha)*torch.sigmoid(self.raw_beta)
        return alpha,beta

    def forward(self,x,state=None):
        B,T,D=x.shape
        q,g,vraw=self.qgv(x).chunk(3,dim=-1)
        gate=g.float().reshape(B,T,self.heads,self.dh)
        v=torch.tanh(vraw.float().reshape(B,T,self.heads,self.dh))
        alpha,beta=self.bounds()
        base=alpha[None,None,:,None]+(beta-alpha)[None,None,:,None]*torch.sigmoid(gate)
        if self.variant=='learned_lm':
            log_stretch=4.0*torch.tanh(self.raw_stretch/4.0)
            inv_stretch=torch.exp(-log_stretch)[None,None,:,None]
            A=torch.exp(torch.log(base)*inv_stretch)
        else:
            A=base
        U=(1.0-A)*v
        E,final=chunk_scan(A,U,state,32)
        z=F.rms_norm(E.reshape(B,T,D),(D,),eps=1e-5)
        y=(torch.sigmoid(q.float())*z).to(x.dtype)
        return self.W_O(y),final

    @torch.no_grad()
    def retention_stats(self):
        a,b=self.bounds()
        if self.variant=='learned_lm':
            stretch=torch.exp(4*torch.tanh(self.raw_stretch/4.0))
        else:
            stretch=torch.ones_like(a)
        lo=a.pow(1.0/stretch)
        hi=b.pow(1.0/stretch)
        # Half-life for the boundary retention rates in token units.
        def half_life(v):return math.log(2)/(-torch.log(v))
        return {'alpha_base':a.cpu().tolist(),'beta_base':b.cpu().tolist(),
                'timescale_factor':stretch.cpu().tolist(),
                'effective_alpha':lo.cpu().tolist(),'effective_beta':hi.cpu().tolist(),
                'half_life_at_alpha':half_life(lo).cpu().tolist(),
                'half_life_at_beta':half_life(hi).cpu().tolist()}


def make_cases(batch_size, length, seed, n_pairs=3, forced_target_position=None,
               with_counterfactual=False):
    """Make one-token-answer associative-recall examples.

    3 random distinct keys, 3 independent random values, 3 fact slots.
    Query targets a uniformly random slot; thus query cannot be ignored.
    A counterfactual swaps ONLY the target value token and answer label.
    Does not depend on a downloaded dataset; identical seeds => paired trials.
    """
    if length < 64:
        raise ValueError('seq len must be at least 64')
    if n_pairs != 3:
        raise ValueError('Current benchmark fixes n_pairs=3 for exact slot matching')
    rng = np.random.default_rng(int(seed))
    x = rng.integers(FILLER_LOW, FILLER_HIGH,
                     size=(batch_size, length), dtype=np.int64)
    labels = np.empty(batch_size, dtype=np.int64)
    alternative = x.copy() if with_counterfactual else None
    alternative_labels = np.empty(batch_size, dtype=np.int64) if with_counterfactual else None
    # All three locations are far from the query, except 'recent'.
    slots = (8, max(30, length//2), length-18)
    if min(slots[1]-slots[0], slots[2]-slots[1]) < 6:
        raise ValueError('Fact insertion locations overlap')
    for b in range(batch_size):
        keys = rng.choice(N_KEYS, 3, replace=False)
        vals = rng.integers(0, N_VALUES, size=3)
        wanted = (rng.integers(0, 3) if forced_target_position is None
                  else int(forced_target_position))
        for slot, key, val in zip(slots, keys, vals):
            x[b, slot:slot+5] = [KEY_MARK, KEY_BASE+int(key), IS_MARK,
                                VALUE_BASE+int(val), END_MARK]
        x[b, -3:] = [QUERY_MARK, KEY_BASE+int(keys[wanted]), ANSWER_MARK]
        labels[b] = VALUE_BASE + int(vals[wanted])
        if with_counterfactual:
            new_val = (int(vals[wanted]) + int(rng.integers(1, N_VALUES))) % N_VALUES
            alternative[b, slots[wanted]+3] = VALUE_BASE + new_val
            alternative_labels[b] = VALUE_BASE + new_val
    return (torch.from_numpy(x), torch.from_numpy(labels),
            torch.from_numpy(alternative) if with_counterfactual else None,
            torch.from_numpy(alternative_labels) if with_counterfactual else None)



def epoch_length(epoch):
    # Gradual extension: 128 -> 256 -> 512; long lengths are held out.
    return [128, 256, 512][min(epoch-1, 2)]


def compute_lr(step, total, base, warmup=100):
    if step < warmup:
        return base*(step+1)/warmup
    progress = min(1.0, max(0., (step-warmup)/max(1, total-warmup)))
    return base*(0.10+0.90*0.5*(1+math.cos(math.pi*progress)))






class KaggleBlock(nn.Module):
    def __init__(self,dim,heads,variant):
        super().__init__()
        self.pre_esa=RMSNorm(dim)
        self.esa=ESAFullMixer(dim,heads,variant)
        self.pre_ffn=RMSNorm(dim)
        self.ffn=nn.Sequential(nn.Linear(dim,4*dim),nn.GELU(),nn.Linear(4*dim,dim))
    def forward(self,x):
        return self.forward_chunk(x,None)[0]
    def forward_chunk(self,x,state=None):
        y,next_state=self.esa(self.pre_esa(x),state)
        x=x+y
        return x+self.ffn(self.pre_ffn(x)),next_state


class KaggleRecall(nn.Module):
    def __init__(self,vocab_size,dim,heads,layers,variant):
        super().__init__()
        self.variant=variant
        self.emb=nn.Embedding(vocab_size,dim)
        self.blocks=nn.ModuleList([KaggleBlock(dim,heads,variant) for _ in range(layers)])
        self.final_norm=RMSNorm(dim)
        self.lm_head=nn.Linear(dim,vocab_size,bias=False)
        self.lm_head.weight=self.emb.weight
    def forward(self,tokens):
        return self.forward_chunk(tokens,None)[0]
    def forward_chunk(self,tokens,states=None):
        x=self.emb(tokens)
        new_states=[]
        for i,blk in enumerate(self.blocks):
            state=None if states is None else states[i]
            x,new_state=blk.forward_chunk(x,state)
            new_states.append(new_state)
        return self.lm_head(self.final_norm(x[:,-1])),new_states
    @torch.no_grad()
    def gate_stats(self):
        return [blk.esa.retention_stats() for blk in self.blocks]


import sys,hashlib,zipfile,shutil

def deterministic_init(model,seed):
    """Same initialization as the current four-model Learned ESA experiment."""
    D=model.emb.embedding_dim
    def normal(shape,tag):
        digest=hashlib.sha256((str(seed)+'|'+tag).encode()).digest()
        gen=torch.Generator(device='cpu').manual_seed(int.from_bytes(digest[:8],'little')%(2**63-1))
        return torch.randn(shape,generator=gen)*.02
    with torch.no_grad():
        model.emb.weight.copy_(normal(model.emb.weight.shape,'embedding'))
        model.final_norm.weight.fill_(1.)
        for i,blk in enumerate(model.blocks):
            blk.pre_esa.weight.fill_(1.)
            blk.pre_ffn.weight.fill_(1.)
            for j,layer in ((0,blk.ffn[0]),(2,blk.ffn[2])):
                layer.weight.copy_(normal(layer.weight.shape,f'ffn-{i}-{j}'))
                layer.bias.zero_()
            mixer=blk.esa
            mixer.W_O.weight.copy_(normal(mixer.W_O.weight.shape,f'out-{i}'))
            g=normal((D,D),f'G-{i}')
            v=normal((D,D),f'V-{i}')
            q=normal((D,D),f'Q-{i}')
            mixer.qgv.weight.copy_(torch.cat([q,g,v]))
    return model


def make_model(args,variant):
    return deterministic_init(KaggleRecall(args.vocab_size,args.dim,args.heads,args.layers,variant),args.seed)


def smoke_test(args):
    torch.set_num_threads(2)
    f=make_model(args,'learned')
    lm=make_model(args,'learned_lm')
    xx=torch.randint(0,args.vocab_size,(2,64))
    with torch.no_grad():
        for a,b in zip(f.parameters(),lm.parameters()):
            # Long memory has one additional vector per layer, so compare state dict by name.
            pass
        for k,v in f.state_dict().items():
            assert torch.equal(v,lm.state_dict()[k]),f'mismatched init {k}'
        y0=f(xx);y1=lm(xx)
        assert torch.allclose(y0,y1,atol=1e-5,rtol=1e-5),('init outputs differ',(y0-y1).abs().max())
        print('PASS identical learned vs learned_lm initialization outputs',float((y0-y1).abs().max()),flush=True)
    for kind,m in [('learned',f),('learned_lm',lm)]:
        m.train();m.zero_grad(set_to_none=True)
        loss=F.cross_entropy(m(xx).float(),torch.randint(VALUE_BASE,VALUE_BASE+N_VALUES,(2,)))
        loss.backward()
        bad=[n for n,p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
        assert not bad,bad
        if kind=='learned_lm':
            for b in m.blocks:
                assert b.esa.raw_stretch.grad is not None
                assert torch.isfinite(b.esa.raw_stretch.grad).all()
        print('PASS',kind,'params',sum(p.numel() for p in m.parameters()),'loss',float(loss),flush=True)
    # Associative streaming equivalence, across 3 successive pieces.
    for mode in ('learned','learned_lm'):
        full_model=make_model(args,mode).eval()
        xall=torch.randint(0,args.vocab_size,(2,96))
        with torch.no_grad():
            full=full_model(xall)
            states=None
            for ch in xall.split(32,dim=1):
                out,states=full_model.forward_chunk(ch,states)
            assert torch.allclose(full,out,rtol=3e-4,atol=3e-4),(mode,(full-out).abs().max())
        m=full_model.blocks[0].esa.eval()
        x=torch.randn(2,96,args.dim)
        with torch.no_grad():
            y,s=m(x)
            ys=[];state=None
            for sub in x.split(32,dim=1):
                out,state=m(sub,state)
                ys.append(out)
            assert torch.allclose(torch.cat(ys,1),y,rtol=2e-4,atol=2e-4)
            assert torch.allclose(s,state,rtol=2e-4,atol=2e-4)
    print('ALL LEARNED ESA / LEARNED LONG-MEMORY TESTS PASSED',flush=True)

@torch.inference_mode()
def evaluate_model(model,device,lengths,trials,seed,eval_batch):
    model.eval();rows=[]
    for length in lengths:
        for pos in range(3):
            total=exact=forced=changed=both=0
            nllsum=0.0
            for start in range(0,trials,eval_batch):
                bs=min(eval_batch,trials-start)
                x,y,x2,y2=make_cases(bs,length,10000000+seed*300001+length*41+pos*1219+start,
                   forced_target_position=pos,with_counterfactual=True)
                paired=torch.cat((x,x2)).to(device)
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                    logits=model(paired).float()
                a,b=logits[:bs],logits[bs:]
                yt=y.to(device);yt2=y2.to(device)
                exact+=int((a.argmax(-1)==yt).sum())
                pred=a[:,VALUE_BASE:VALUE_BASE+N_VALUES].argmax(-1)+VALUE_BASE
                pred2=b[:,VALUE_BASE:VALUE_BASE+N_VALUES].argmax(-1)+VALUE_BASE
                forced+=int((pred==yt).sum())
                changed+=int((pred!=pred2).sum())
                both+=int(((pred==yt)&(pred2==yt2)).sum())
                nllsum+=float(F.cross_entropy(a,yt,reduction='sum'))
                total+=bs
            rows.append({'length':length,'position':['early','middle','recent'][pos],
                'trials':total,'exact_accuracy':exact/total,'forced_accuracy':forced/total,
                'counterfactual_change':changed/total,'paired_both_correct':both/total,
                'answer_nll':nllsum/total,'answer_ppl':math.exp(min(50,nllsum/total))})
    return rows



def settings_hash(args,variant):
    settings={k:getattr(args,k) for k in ('steps_per_epoch','epochs','batch_size','dim',
        'heads','layers','vocab_size','lr','seed','max_context','eval_trials',
        'final_eval_trials','eval_batch')}
    settings['variant']=variant
    settings['architecture']='learned_bounded_write_timescale_v1'
    settings['worker_sha256']='see source archive'
    return hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()

def start_model(args,variant,device):
    folder=Path(args.root)/variant;folder.mkdir(parents=True,exist_ok=True)
    model=make_model(args,variant).to(device)
    regular=[];special=[]
    for n,p in model.named_parameters():
        (special if n.endswith(('raw_alpha','raw_beta','raw_stretch')) else regular).append(p)
    optimizer=torch.optim.AdamW([{'params':regular,'weight_decay':.1},
                                 {'params':special,'weight_decay':0.}],lr=args.lr)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda',
                                 init_scale=1024.0,growth_interval=2000)
    task=dict(variant=variant,folder=folder,model=model,optimizer=optimizer,
        scaler=scaler,step=0,loss_total=0.,updates=0,seconds=0.,
        amp_overflows=0,fp32_fallbacks=0,config_hash=settings_hash(args,variant))
    ck=folder/'latest.pt'
    if args.resume and ck.exists():
        data=torch.load(ck,map_location='cpu',weights_only=False)
        if data['config_hash']!=task['config_hash']:raise ValueError(f'Config changed, cannot resume {variant}')
        model.load_state_dict(data['model'],strict=True)
        optimizer.load_state_dict(data['optimizer']);scaler.load_state_dict(data['scaler'])
        task.update({k:data[k] for k in ('step','loss_total','updates','seconds')})
        task['amp_overflows']=int(data.get('amp_overflows',0))
        task['fp32_fallbacks']=int(data.get('fp32_fallbacks',0))
        print('RESUMED',variant,task['step'],flush=True)
    elif ck.exists():
        raise RuntimeError(f'Checkpoint already present: {ck}; use --resume to avoid overwriting')
    if not (folder/'metrics.csv').exists():
        with (folder/'metrics.csv').open('w',newline='') as f:
            csv.writer(f).writerow(['variant','epoch','step','context','mean_epoch_train_loss',
                'exact512','forced512','nll512','train_tok_s','peak_vram_gib','parameters'])
    print('INITIALIZED',variant,'params',sum(p.numel() for p in model.parameters()),
          'step',task['step'],flush=True)
    return task

def save_task(t):
    d={k:t[k] for k in ('step','loss_total','updates','seconds','config_hash',
                       'variant','amp_overflows','fp32_fallbacks')}
    d.update(model=t['model'].state_dict(),optimizer=t['optimizer'].state_dict(),
             scaler=t['scaler'].state_dict(),format_version=2)
    temp=t['folder']/'latest.pt.tmp';torch.save(d,temp);os.replace(temp,t['folder']/'latest.pt')
    print('CHECKPOINT',t['variant'],t['step'],flush=True)

def epoch_report(args,t,device):
    ep=t['step']//args.steps_per_epoch
    trials=args.final_eval_trials if ep==args.epochs else args.eval_trials
    rows=evaluate_model(t['model'],device,[128,512,1024,2048,4096],trials,args.seed,args.eval_batch)
    path=t['folder']/f'eval_epoch{ep}.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    r512=[r for r in rows if r['length']==512]
    exact=sum(v['exact_accuracy'] for v in r512)/3
    forced=sum(v['forced_accuracy'] for v in r512)/3
    nll=sum(v['answer_nll'] for v in r512)/3
    length=epoch_length(ep)
    rate=t['updates']*args.batch_size*length/max(1e-9,t['seconds'])
    peak=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else 0.
    with (t['folder']/'metrics.csv').open('a',newline='') as f:
        csv.writer(f).writerow([t['variant'],ep,t['step'],length,
            t['loss_total']/max(1,t['updates']),exact,forced,nll,rate,peak,
            sum(p.numel() for p in t['model'].parameters())])
    (t['folder']/f'gate_stats_epoch{ep}.json').write_text(json.dumps(t['model'].gate_stats(),indent=2))
    t.update(loss_total=0.,updates=0,seconds=0.)
    save_task(t)
    print(f'EPOCH_RESULT {t["variant"]} {ep} EXACT512={exact:.2%} NLL512={nll:.4f} RATE={rate:.0f}',flush=True)

def train_step(args,t,device):
    """One SUCCESSFUL optimizer update. Automatically recover from normal AMP overflow.

    FP16 gradients may overflow when GradScaler's scale is too large. Such events
    are *not* non-finite FP32 model gradients. Let GradScaler skip the update and
    decrease scale, then retry the SAME batch without advancing the global step.
    If repeated scaling reductions cannot help, do one FP32 step and report it.
    """
    gstep=t['step'];ep=gstep//args.steps_per_epoch+1
    length=epoch_length(ep)
    x,y,_,_=make_cases(args.batch_size,length,args.seed*1000009+gstep*7919)
    x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
    model=t['model'];opt=t['optimizer'];scaler=t['scaler']
    lr=compute_lr(gstep,args.steps_per_epoch*args.epochs,args.lr)
    for group in opt.param_groups:group['lr']=lr
    model.train()
    start=time.perf_counter()
    success=False
    MAX_AMP_RETRIES=12
    # Each retry uses the same training sample and does not change the optimizer
    # parameters. PyTorch GradScaler skips a step if it observes nonfinite grads.
    for retry in range(MAX_AMP_RETRIES if device.type=='cuda' else 1):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,dtype=torch.float16,
                            enabled=device.type=='cuda'):
            logits=model(x)
            loss=F.cross_entropy(logits.float(),y)
        if not torch.isfinite(loss):
            # This is different from backward overflow: the forward loss itself
            # is nonfinite. Do not silently treat a broken model as an AMP skip.
            raise FloatingPointError(f'Nonfinite forward loss {t["variant"]} '
                                     f'at step {gstep} (AMP scale {scaler.get_scale()})')
        if device.type=='cuda':
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            # Detect actual nonfinite gradient elements before clipping them.
            bad=[]
            for n,p in model.named_parameters():
                if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                    bad.append(n)
            if bad:
                old=float(scaler.get_scale())
                # GradScaler's found_inf flag was recorded by unscale_().
                # scaler.step() will skip the optimizer step, then update()
                # reduces the scaling factor. There is no weight update here.
                scaler.step(opt)
                scaler.update()
                new=float(scaler.get_scale())
                t['amp_overflows']+=1
                print(f'AMP_OVERFLOW_RECOVER {t["variant"]} step={gstep} '
                      f'retry={retry+1} scale={old:g}->{new:g} '
                      f'bad_grads={bad[:4]}',flush=True)
                if new>=old:
                    raise FloatingPointError('GradScaler did not reduce scale on '
                        f'nonfinite gradients ({t["variant"]}, step={gstep}).')
                continue
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError('Nonfinite total FP32 gradient norm '
                    f'with individually finite grads for {t["variant"]} at {gstep}')
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError(f'Nonfinite true CPU gradient {t["variant"]} at {gstep}')
            opt.step()
        success=True
        break
    if not success:
        # Rare fallback: scaler cannot find a safe FP16 backward at this step.
        # Compute one full FP32 forward/backward on the SAME training batch.
        # Log and count it: this may affect speed but does not change math.
        t['fp32_fallbacks']+=1
        print(f'AMP_FP32_FALLBACK {t["variant"]} step={gstep} '
              f'after={MAX_AMP_RETRIES} overflows',flush=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,enabled=False):
            logits=model(x)
            loss=F.cross_entropy(logits.float(),y)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f'Nonfinite genuine FP32 forward loss '
                                     f'{t["variant"]} at step {gstep}')
        loss.backward()
        bad=[n for n,p in model.named_parameters()
             if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
        if bad:
            raise FloatingPointError(f'Nonfinite genuine FP32 gradients for '
                                     f'{t["variant"]} step={gstep}: {bad[:8]}')
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError(f'Nonfinite genuine FP32 norm '
                                     f'{t["variant"]} at step {gstep}')
        opt.step()
    if device.type=='cuda':torch.cuda.synchronize()
    t['seconds']+=time.perf_counter()-start
    t['loss_total']+=float(loss.detach().item())
    t['updates']+=1;t['step']+=1
    if t['step']%args.log_every==0:
        print(f'PROGRESS {t["variant"]} epoch={ep}/{args.epochs} '
              f'step={t["step"]}/{args.steps_per_epoch*args.epochs} '
              f'mean_epoch_loss={t["loss_total"]/max(1,t["updates"]):.4f} '
              f'amp_overflows={t["amp_overflows"]} '
              f'fp32_fallbacks={t["fp32_fallbacks"]}',flush=True)
    if t['step']%args.save_every==0:save_task(t)
    if t['step']%args.steps_per_epoch==0:epoch_report(args,t,device)

@torch.inference_mode()
def speed_benchmark(args):
    """Measure *full six-layer model* prefill and state-cached autoregressive decode.

    Same input shapes, precision, warmup, repeats and environment for both models.
    CUDA sync brackets measure true elapsed GPU completion. Decode timings
    include Python/PyTorch launch overhead (no custom fused production kernel).
    """
    device=torch.device('cuda:0' if args.device=='cuda' else 'cpu')
    model=make_model(args,args.variants).to(device).eval()
    folder=Path(args.root)/args.variants
    ck=folder/'latest.pt'
    if not ck.exists():raise FileNotFoundError(f'Missing trained checkpoint: {ck}')
    data=torch.load(ck,map_location='cpu',weights_only=False)
    if int(data['step'])!=args.epochs*args.steps_per_epoch:
        raise RuntimeError(f'Expected 60K trained weights but found {data["step"]} in {ck}')
    if data['config_hash']!=settings_hash(args,args.variants):
        raise RuntimeError('Model config hash mismatch; speed and training configs must match')
    model.load_state_dict(data['model'],strict=True)
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    rows=[]
    for length in (128,512,1024,2048,4096):
        ids=torch.randint(FILLER_LOW,FILLER_HIGH,(1,length),device=device)
        def full():
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                return model(ids)
        full();full();sync()
        t0=time.perf_counter()
        n=5
        for _ in range(n):full()
        sync();elapsed=time.perf_counter()-t0
        pre_ms=elapsed/n*1e3
        # Build cached state on the SAME history with all layers, in 128-token chunks.
        state=None
        for prefix in ids.split(128,dim=1):
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                _,state=model.forward_chunk(prefix,state)
        sync()
        toks=torch.randint(FILLER_LOW,FILLER_HIGH,(1,args.decode_tokens),device=device)
        tt=time.perf_counter()
        for q in range(args.decode_tokens):
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                _,state=model.forward_chunk(toks[:,q:q+1],state)
        sync();de_ms=(time.perf_counter()-tt)/args.decode_tokens*1e3
        rows.append(dict(model=args.variants,checkpoint_step=int(data['step']),context=length,
                         prefill_ms=pre_ms,prefill_tokens_per_second=length/(pre_ms/1e3),
                         cached_decode_ms_per_token=de_ms,cached_decode_tokens_per_second=1e3/de_ms,
                         parameters=sum(p.numel() for p in model.parameters()),
                         dtype='amp_fp16' if device.type=='cuda' else 'fp32'))
        print('SPEED',args.variants,length,round(pre_ms,3),'ms prefill',
              round(de_ms,4),'ms/token decode',flush=True)
    output=folder/'speed.csv'
    with output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('SPEED_SAVED',output,flush=True)


def worker(args):
    device=torch.device('cuda:0' if args.device=='cuda' else 'cpu')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Select Kaggle GPU T4 x2')
    torch.set_num_threads(2)
    if device.type=='cuda':
        print('GPU',torch.cuda.get_device_name(0),'VRAM_GiB',torch.cuda.get_device_properties(0).total_memory/2**30,flush=True)
    variants=args.variants.split(',')
    if len(variants)!=1:raise ValueError('One model per GPU worker is required')
    tasks=[start_model(args,v,device) for v in variants]
    # Recover a checkpoint written at an epoch boundary before evaluation finished.
    for t in tasks:
        if (t['step']>0 and t['step']%args.steps_per_epoch==0 and t['updates']>0):
            print('REPAIR_INTERRUPTED_EPOCH_EVAL',t['variant'],t['step'],flush=True)
            epoch_report(args,t,device)
    total=args.epochs*args.steps_per_epoch
    start_steps={t['variant']:t['step'] for t in tasks}
    while any(t['step']<total for t in tasks):
        progress=False
        for t in tasks:
            if t['step']<total and (not args.max_steps_this_run or
                t['step']-start_steps[t['variant']]<args.max_steps_this_run):
                train_step(args,t,device);progress=True
        if not progress:break
    for t in tasks:save_task(t)
    print('WORKER_DONE',[(t['variant'],t['step'],t['amp_overflows'],
                          t['fp32_fallbacks']) for t in tasks],flush=True)

def parse():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/kaggle/working/MLBricks_Learned_ESA_vs_LearnedLM_60K')
    p.add_argument('--variants',default='learned')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--steps-per-epoch',type=int,default=20000)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--dim',type=int,default=384)
    p.add_argument('--heads',type=int,default=6)
    p.add_argument('--layers',type=int,default=6)
    p.add_argument('--vocab-size',type=int,default=256)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-context',type=int,default=4096)
    p.add_argument('--eval-trials',type=int,default=12)
    p.add_argument('--final-eval-trials',type=int,default=170)
    p.add_argument('--eval-batch',type=int,default=1)
    p.add_argument('--save-every',type=int,default=500)
    p.add_argument('--log-every',type=int,default=200)
    p.add_argument('--max-steps-this-run',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--speed',action='store_true')
    p.add_argument('--decode-tokens',type=int,default=64)
    args=p.parse_args()
    assert args.dim%args.heads==0 and args.heads==6 and args.max_context>=4096
    return args

if __name__=='__main__':
    a=parse()
    if a.smoke:smoke_test(a)
    elif a.speed:speed_benchmark(a)
    else:worker(a)

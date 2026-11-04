
#!/usr/bin/env python3
"""Original 60K ESA Long-Memory vs bounded-write/query-read hybrid.

The `original` mixer is the mathematical model from
MLBricks_ESA_Long_Memory_Dual_T4_60K_Resumable.ipynb (2026):
  A=exp(-ln(2)*softplus(W_A*x+b_h)/exp(ell_h)); V=W_V*x;
  E=A*Eprev+V; Y=W_O*RMSNorm(E).

`hybrid` changes ONLY mixer value/write/read/projection:
  [Q,G,V_raw]=W_QGV*x; A=same half-life gate;
  E=A*Eprev+(1-A)*tanh(V_raw);
  Y=W_O(sigmoid(Q)*RMSNorm(E)).

Same RMSNorm/FFN/embedding/readout scaffolding, data, optimizer,
step-indexed examples, and evaluation. This is SYNTHETIC recall, NOT
natural-language perplexity. Both models start with matched shared weights.
No Triton, internet, external code or CUDA extension builds.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

KEY_MARK, IS_MARK, END_MARK, QUERY_MARK, ANSWER_MARK = 2,3,4,5,6
KEY_BASE, N_KEYS = 16, 32
VALUE_BASE, N_VALUES = 64, 32
FILLER_LOW, FILLER_HIGH = 128,256
INITIAL_HALF_LIVES=(16.,64.,256.,1024.,4096.,8192.)

def initial_half_life_logs(heads):
    """Keep original six-head initialization; interpolate in log space otherwise."""
    values = torch.tensor(INITIAL_HALF_LIVES).log()
    if heads == len(INITIAL_HALF_LIVES):
        return values
    return torch.linspace(float(values[0]), float(values[-1]), int(heads))

ARCH_VERSION='original_esa_lm_vs_bounded_hybrid_v1_2026_11'


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

class RMSNorm(nn.Module):
    def __init__(self,dim,eps=1e-5):
        super().__init__(); self.weight=nn.Parameter(torch.ones(dim)); self.eps=eps
    def forward(self,x):return F.rms_norm(x,(x.shape[-1],),self.weight.to(x.dtype),self.eps)


def chunk_scan(a,v,initial=None,chunk_size=32):
    """Division-free FP32 associative affine scan, preserving E_t=A_t*E_prev+U_t.

    A prefix (A,B) represents a map e -> A*e+B.  Pair composition is:
       (A_b,B_b) o (A_a,B_a) = (A_b*A_a, B_b+A_b*B_a).
    Hillis-Steele doubling needs O(log(chunk_size)) parallel stages, never
    divides by cumprod(A); this avoids dangerous forward/backward 1/prod(A).
    """
    B,T,H,Dh=a.shape
    state=v.new_zeros(B,H,Dh).float() if initial is None else initial.float()
    parts=[]
    for start in range(0,T,chunk_size):
        aa=a[:,start:start+chunk_size].float()
        bb=v[:,start:start+chunk_size].float()
        distance=1
        while distance < aa.shape[1]:
            prev_a,prev_b=aa,bb
            aa=torch.cat([prev_a[:,:distance],
                          prev_a[:,distance:]*prev_a[:,:-distance]],dim=1)
            bb=torch.cat([prev_b[:,:distance],
                          prev_b[:,distance:]+prev_a[:,distance:]*prev_b[:,:-distance]],dim=1)
            distance*=2
        ss=bb+aa*state[:,None]
        state=ss[:,-1]
        parts.append(ss)
    return torch.cat(parts,dim=1),state


def stable_clip_grad_norm_(parameters,max_norm=1.0):
    """Clip in float64 to avoid overflow in the FP32 aggregate L2 norm.

    Loss-scaler overflows must be handled BEFORE this function. This function
    does not replace inf/nan gradients with zeros, and never silently accepts
    a non-finite gradient.
    """
    grads=[p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    total_sq=torch.zeros((),device=grads[0].device,dtype=torch.float64)
    for grad in grads:
        if not bool(torch.isfinite(grad).all()):
            raise FloatingPointError('A gradient remains nonfinite after AMP unscale')
        total_sq=total_sq+grad.detach().to(torch.float64).square().sum()
    norm=total_sq.sqrt()
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError('Total gradient norm is nonfinite even in float64')
    coeff=min(1.0,float(max_norm)/max(1e-30,float(norm)))
    if coeff<1.0:
        for grad in grads:
            grad.mul_(coeff)
    return float(norm)


class OriginalESA(nn.Module):
    """Unchanged mathematical mixer from completed original ESA-LM 60K study."""
    def __init__(self,dim,heads,chunk_size=32):
        super().__init__()
        assert dim%heads==0 and heads>0
        self.dim,self.heads,self.dh,self.chunk_size=dim,heads,dim//heads,chunk_size
        self.W_A=nn.Linear(dim,dim,bias=False)
        self.W_V=nn.Linear(dim,dim,bias=False)
        self.W_O=nn.Linear(dim,dim,bias=False)
        self.state_norm=RMSNorm(dim)
        self.log_half_life=nn.Parameter(initial_half_life_logs(heads))
        self.raw_decay_bias=nn.Parameter(torch.full((heads,),math.log(math.expm1(1.))))
    def forward(self,x,state=None):
        B,T,_=x.shape
        z=self.W_A(x).reshape(B,T,self.heads,self.dh).float()
        half_life=self.log_half_life.float().exp()
        decay=F.softplus(z+self.raw_decay_bias[None,None,:,None])
        a=torch.exp(-math.log(2.)*decay/half_life[None,None,:,None])
        v=self.W_V(x).reshape(B,T,self.heads,self.dh)
        s,last=chunk_scan(a,v,state,self.chunk_size)
        y=self.state_norm(s.reshape(B,T,self.dim)).to(x.dtype)
        return self.W_O(y),last
    def retention_stats(self):return half_life_stats(self)


class HybridESA(nn.Module):
    """Same half-life A, but bounded writes and gated query readout."""
    def __init__(self,dim,heads,chunk_size=32):
        super().__init__()
        assert dim%heads==0 and heads>0
        self.dim,self.heads,self.dh,self.chunk_size=dim,heads,dim//heads,chunk_size
        self.qgv=nn.Linear(dim,3*dim,bias=False)
        self.W_O=nn.Linear(dim,dim,bias=False)
        self.state_norm=RMSNorm(dim)
        self.log_half_life=nn.Parameter(initial_half_life_logs(heads))
        self.raw_decay_bias=nn.Parameter(torch.full((heads,),math.log(math.expm1(1.))))
    def forward(self,x,state=None):
        B,T,_=x.shape
        q,g,raw_v=self.qgv(x).chunk(3,dim=-1)
        z=g.reshape(B,T,self.heads,self.dh).float()
        half_life=self.log_half_life.float().exp()
        decay=F.softplus(z+self.raw_decay_bias[None,None,:,None])
        a=torch.exp(-math.log(2.)*decay/half_life[None,None,:,None])
        v=torch.tanh(raw_v.reshape(B,T,self.heads,self.dh).float())
        u=(1.0-a)*v
        s,last=chunk_scan(a,u,state,self.chunk_size)
        norm=self.state_norm(s.reshape(B,T,self.dim))
        read=torch.sigmoid(q.float())*norm
        return self.W_O(read.to(x.dtype)),last
    def retention_stats(self):return half_life_stats(self)

@torch.no_grad()
def half_life_stats(m):
    half=m.log_half_life.exp()
    bias=m.raw_decay_bias
    typical_decay=F.softplus(bias)
    # At z=0: actual A=0.5**(typical_decay/half_life)
    return dict(half_life_parameter=half.detach().cpu().tolist(),
                decay_bias=bias.detach().cpu().tolist(),
                effective_half_life_at_zero_gate=(half/typical_decay).detach().cpu().tolist())


class DecoderBlock(nn.Module):
    def __init__(self,dim,heads,variant,chunk_size):
        super().__init__()
        self.pre_esa=RMSNorm(dim)
        self.esa=(OriginalESA if variant=='original' else HybridESA)(dim,heads,chunk_size)
        self.pre_ffn=RMSNorm(dim)
        self.ffn=nn.Sequential(nn.Linear(dim,4*dim),nn.GELU(),nn.Linear(4*dim,dim))
    def forward(self,x,state=None):
        y,new_state=self.esa(self.pre_esa(x),state)
        x=x+y
        return x+self.ffn(self.pre_ffn(x)),new_state


class RecallModel(nn.Module):
    def __init__(self,vocab_size,dim,heads,layers,variant,chunk_size):
        super().__init__()
        self.emb=nn.Embedding(vocab_size,dim)
        self.blocks=nn.ModuleList([DecoderBlock(dim,heads,variant,chunk_size) for _ in range(layers)])
        self.final_norm=RMSNorm(dim)
        self.lm_head=nn.Linear(dim,vocab_size,bias=False)
        self.lm_head.weight=self.emb.weight
    def forward(self,tokens,cached_states=None):
        x=self.emb(tokens)
        states=[]
        for i,blk in enumerate(self.blocks):
            last=None if cached_states is None else cached_states[i]
            x,s=blk(x,last);states.append(s)
        return self.lm_head(self.final_norm(x[:,-1,:])),states
    @torch.no_grad()
    def all_stats(self):return [b.esa.retention_stats() for b in self.blocks]


def seeded_normal(seed,tag,shape):
    digest=hashlib.sha256((str(seed)+'|'+tag).encode()).digest()
    g=torch.Generator(device='cpu').manual_seed(int.from_bytes(digest[:8],'little')%(2**63-1))
    return .02*torch.randn(tuple(shape),generator=g)


def build_model(args,variant):
    """Shared weights are identical for every matching component.

    Original's W_A equals hybrid's packed G slice; original W_V equals
    hybrid's V slice. Q is the only extra matrix and has a distinct seed.
    The initial outputs will differ due to architecture, by design.
    """
    model=RecallModel(args.vocab_size,args.dim,args.heads,args.layers,variant,args.chunk_size)
    D=args.dim
    with torch.no_grad():
        model.emb.weight.copy_(seeded_normal(args.seed,'embedding',model.emb.weight.shape))
        model.final_norm.weight.fill_(1.)
        for i,b in enumerate(model.blocks):
            b.pre_esa.weight.fill_(1.);b.pre_ffn.weight.fill_(1.)
            for j in [0,2]:
                b.ffn[j].weight.copy_(seeded_normal(args.seed,f'ffn-{i}-{j}',b.ffn[j].weight.shape))
                b.ffn[j].bias.zero_()
            e=b.esa;e.state_norm.weight.fill_(1.)
            e.W_O.weight.copy_(seeded_normal(args.seed,f'out-{i}',e.W_O.weight.shape))
            ga=seeded_normal(args.seed,f'gate-{i}',(D,D))
            vv=seeded_normal(args.seed,f'value-{i}',(D,D))
            if variant=='original':e.W_A.weight.copy_(ga);e.W_V.weight.copy_(vv)
            else:
                qq=seeded_normal(args.seed,f'query-{i}',(D,D))
                e.qgv.weight.copy_(torch.cat([qq,ga,vv],dim=0))
    return model


def make_cases(batch_size,length,seed,n_pairs=3,forced_target_position=None,
               with_counterfactual=False):
    """EXACT deterministic synthetic task from original ESA-LM notebook."""
    if length<64:raise ValueError('length must be >=64')
    if n_pairs!=3:raise ValueError('n_pairs must be 3')
    rng=np.random.default_rng(int(seed))
    x=rng.integers(FILLER_LOW,FILLER_HIGH,size=(batch_size,length),dtype=np.int64)
    labels=np.empty(batch_size,dtype=np.int64)
    alternative=x.copy() if with_counterfactual else None
    alternative_labels=np.empty(batch_size,dtype=np.int64) if with_counterfactual else None
    slots=(8,max(30,length//2),length-18)
    if min(slots[1]-slots[0],slots[2]-slots[1])<6:raise ValueError('overlapping facts')
    for b in range(batch_size):
        keys=rng.choice(N_KEYS,3,replace=False)
        vals=rng.integers(0,N_VALUES,size=3)
        wanted=(rng.integers(0,3) if forced_target_position is None
                else int(forced_target_position))
        for slot,key,val in zip(slots,keys,vals):
            x[b,slot:slot+5]=[KEY_MARK,KEY_BASE+int(key),IS_MARK,VALUE_BASE+int(val),END_MARK]
        x[b,-3:]=[QUERY_MARK,KEY_BASE+int(keys[wanted]),ANSWER_MARK]
        labels[b]=VALUE_BASE+int(vals[wanted])
        if with_counterfactual:
            # Clone the completed factual sequence before swapping ONLY its value.
            # The original notebook copied filler before injecting facts; that
            # invalidated paired-counterfactual measurements. This fixes only x2,
            # leaving original x, labels, seeds and ordinary recall unchanged.
            alternative[b] = x[b]
            new_val=(int(vals[wanted])+int(rng.integers(1,N_VALUES)))%N_VALUES
            alternative[b,slots[wanted]+3]=VALUE_BASE+new_val
            alternative_labels[b]=VALUE_BASE+new_val
    return (torch.from_numpy(x),torch.from_numpy(labels),
            torch.from_numpy(alternative) if with_counterfactual else None,
            torch.from_numpy(alternative_labels) if with_counterfactual else None)


def epoch_length(epoch):return [128,256,512][min(epoch-1,2)]

def learning_rate(step,total,base,warmup=100):
    if step<warmup:return base*(step+1)/warmup
    progress=min(1.,max(0.,(step-warmup)/max(1,total-warmup)))
    return base*(.10+.90*.5*(1+math.cos(math.pi*progress)))


def run_logits(model,x,device,chunk=128):
    states=None; logits=None
    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=(device.type=='cuda')):
        for start in range(0,x.shape[1],chunk):
            logits,states=model(x[:,start:start+chunk].to(device),states)
    return logits.float()

@torch.inference_mode()
def evaluate(model,device,lengths,trials,seed,eval_batch=3):
    model.eval();rows=[]
    for length in lengths:
        for pos in range(3):
            total=exact=forced=changed=both=0;nll_sum=0.
            for offset in range(0,trials,eval_batch):
                bs=min(trials-offset,eval_batch)
                x,y,x2,y2=make_cases(bs,length,10000000+seed*300001+length*41+pos*1219+offset,
                              forced_target_position=pos,with_counterfactual=True)
                logits=run_logits(model,torch.cat((x,x2)),device)
                a,b=logits[:bs],logits[bs:]
                target=y.to(device); target2=y2.to(device)
                exact+=int((a.argmax(-1)==target).sum())
                av=a[:,VALUE_BASE:VALUE_BASE+N_VALUES].argmax(-1)+VALUE_BASE
                bv=b[:,VALUE_BASE:VALUE_BASE+N_VALUES].argmax(-1)+VALUE_BASE
                forced+=int((av==target).sum())
                changed+=int((av!=bv).sum())
                both+=int(((av==target)&(bv==target2)).sum())
                nll_sum+=float(F.cross_entropy(a,target,reduction='sum'))
                total+=bs
            nll=nll_sum/total
            rows.append(dict(length=length,position=('early','middle','recent')[pos],
                             trials=total,exact_accuracy=exact/total,
                             forced_accuracy=forced/total,
                             counterfactual_change=changed/total,
                             paired_both_correct=both/total,target_nll=nll,
                             answer_ppl=math.exp(min(50.,nll)),chance=1/N_VALUES))
    return rows


def summary_eval(rows):
    n=sum(r['trials'] for r in rows)
    return {'cases':n,'exact_accuracy':sum(r['trials']*r['exact_accuracy'] for r in rows)/n,
            'forced_accuracy':sum(r['trials']*r['forced_accuracy'] for r in rows)/n,
            'target_nll':sum(r['trials']*r['target_nll'] for r in rows)/n,
            'answer_ppl':math.exp(min(50.,sum(r['trials']*r['target_nll'] for r in rows)/n)),
            'paired_both_correct':sum(r['trials']*r['paired_both_correct'] for r in rows)/n,
            'counterfactual_change':sum(r['trials']*r['counterfactual_change'] for r in rows)/n}


def atomic_json(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2))
    os.replace(tmp,path)


def config_hash(args,variant):
    keys=('epochs','steps_per_epoch','batch_size','dim','heads','layers','vocab_size',
          'lr','seed','chunk_size','eval_trials','final_eval_trials','eval_batch')
    return hashlib.sha256(json.dumps({'architecture':ARCH_VERSION,'variant':variant,
                  **{k:getattr(args,k) for k in keys}},sort_keys=True).encode()).hexdigest()


def load_checkpoint(path):
    return torch.load(path,map_location='cpu',weights_only=False)


def save_checkpoint(task):
    obj={k:task[k] for k in ('variant','step','loss_sum','loss_count','epoch_seconds',
           'overflows','fp32_fallbacks','config_hash','last_evaluated')}
    obj.update(version=1,model=task['model'].state_dict(),optimizer=task['optimizer'].state_dict(),
               scaler=task['scaler'].state_dict(),rng=torch.get_rng_state(),
               python_rng=random.getstate(),numpy_rng=np.random.get_state(),
               cuda_rng=torch.cuda.get_rng_state_all() if task['device'].type=='cuda' else None)
    target=task['folder']/'latest.pt';tmp=target.with_suffix('.pt.tmp')
    torch.save(obj,tmp);os.replace(tmp,target)
    print('CHECKPOINT',task['variant'],task['step'],flush=True)


def make_task(args,variant,device):
    folder=Path(args.root)/variant;folder.mkdir(parents=True,exist_ok=True)
    model=build_model(args,variant).to(device)
    regular=[];special=[]
    for name,param in model.named_parameters():
        (special if name.endswith(('log_half_life','raw_decay_bias')) else regular).append(param)
    optimizer=torch.optim.AdamW([{'params':regular,'weight_decay':.1},
                                 {'params':special,'weight_decay':0.}],lr=args.lr)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda',
                                init_scale=1024.,growth_interval=2000)
    task=dict(variant=variant,model=model,optimizer=optimizer,scaler=scaler,
              folder=folder,device=device,step=0,loss_sum=0.,loss_count=0,
              epoch_seconds=0.,overflows=0,fp32_fallbacks=0,last_evaluated=0,config_hash=config_hash(args,variant))
    path=folder/'latest.pt'
    if path.exists():
        if not args.resume:raise RuntimeError(f'{path} exists; --resume to continue safely')
        obj=load_checkpoint(path)
        if obj['config_hash']!=task['config_hash'] or obj['variant']!=variant:
            raise RuntimeError('checkpoint configuration mismatch; do not silently overwrite')
        model.load_state_dict(obj['model'],strict=True)
        optimizer.load_state_dict(obj['optimizer']);scaler.load_state_dict(obj['scaler'])
        for k in ('step','loss_sum','loss_count','epoch_seconds','overflows','fp32_fallbacks','last_evaluated'):
            task[k]=obj.get(k,0)
        torch.set_rng_state(obj['rng']);random.setstate(obj['python_rng']);np.random.set_state(obj['numpy_rng'])
        if device.type=='cuda' and obj.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(obj['cuda_rng'])
        print('RESUMED',variant,'step',task['step'],flush=True)
    else:
        print('INITIALIZED',variant,'params',sum(p.numel() for p in model.parameters()),flush=True)
    return task


def training_update(args,task):
    device=task['device'];model=task['model'];opt=task['optimizer'];scaler=task['scaler']
    step=task['step'];epoch=step//args.steps_per_epoch+1
    length=epoch_length(epoch)
    x,y,_,_=make_cases(args.batch_size,length,args.seed*1000009+step*7919)
    x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
    for group in opt.param_groups:
        group['lr']=learning_rate(step,args.epochs*args.steps_per_epoch,args.lr)
    model.train();started=time.perf_counter()
    completed=False;loss=None
    for retry in range(12 if device.type=='cuda' else 1):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=(device.type=='cuda')):
            logits,_=model(x)
            loss=F.cross_entropy(logits.float(),y)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f'Nonfinite FORWARD loss {task["variant"]} at {step+1}')
        if device.type=='cuda':
            scaler.scale(loss).backward();scaler.unscale_(opt)
            bad=[n for n,p in model.named_parameters() if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
            if bad:
                old=float(scaler.get_scale())
                scaler.step(opt)  # skip update because GradScaler observed inf
                scaler.update();new=float(scaler.get_scale())
                task['overflows']+=1
                print(f'AMP_OVERFLOW_RECOVER {task["variant"]} step={step+1} retry={retry+1} '
                      f'scale={old:g}->{new:g} bad_grads={bad[:3]}',flush=True)
                if new>=old:raise RuntimeError('GradScaler failed to lower scale')
                continue
        else:
            loss.backward()
        norm=stable_clip_grad_norm_(model.parameters(),1.)
        if norm>1.e5:
            print(f'LARGE_PRECLIP_GRAD {task["variant"]} step={step+1} L2={norm:.4g}',flush=True)
        if device.type=='cuda':scaler.step(opt);scaler.update()
        else:opt.step()
        completed=True;break
    if not completed:
        # No successful optimizer update under AMP; use true FP32 once.
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,enabled=False):
            logits,_=model(x);loss=F.cross_entropy(logits.float(),y)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f'FP32 forward loss nonfinite {task["variant"]} at {step+1}')
        loss.backward()
        norm=stable_clip_grad_norm_(model.parameters(),1.)
        if norm>1.e5:
            print(f'LARGE_PRECLIP_GRAD_FP32 {task["variant"]} step={step+1} L2={norm:.4g}',flush=True)
        opt.step();task['fp32_fallbacks']+=1
        print('FP32_FALLBACK',task['variant'],'step',step+1,flush=True)
    if device.type=='cuda':torch.cuda.synchronize()
    task['step']+=1;task['loss_sum']+=float(loss.detach());task['loss_count']+=1
    task['epoch_seconds']+=time.perf_counter()-started
    if task['step']%args.log_every==0:
        print(f'PROGRESS {task["variant"]} epoch={epoch}/{args.epochs} '
              f'step={task["step"]}/{args.epochs*args.steps_per_epoch} '
              f'mean_epoch_loss={task["loss_sum"]/task["loss_count"]:.4f} '
              f'amp_overflows={task["overflows"]} fp32_fallbacks={task["fp32_fallbacks"]}',flush=True)


def epoch_evaluation(args,task):
    step=task['step'];epoch=step//args.steps_per_epoch
    trials=args.final_eval_trials if epoch==args.epochs else args.eval_trials
    model=task['model'];device=task['device'];folder=task['folder']
    records=evaluate(model,device,[128,512,1024,2048,4096],trials,args.seed,args.eval_batch)
    dest=folder/f'eval_epoch{epoch}.csv'
    tmp=dest.with_suffix('.csv.tmp')
    with tmp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    os.replace(tmp,dest)
    overall=summary_eval(records)
    r512=summary_eval([r for r in records if r['length']==512])
    speed=task['loss_count']*args.batch_size*epoch_length(epoch)/max(1e-9,task['epoch_seconds'])
    gpu_gib=(torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else 0.)
    row=dict(model=task['variant'],epoch=epoch,step=step,train_length=epoch_length(epoch),
             mean_epoch_train_loss=task['loss_sum']/max(1,task['loss_count']),
             exact_accuracy=overall['exact_accuracy'],answer_nll=overall['target_nll'],
             answer_ppl=overall['answer_ppl'],forced_accuracy=overall['forced_accuracy'],
             counterfactual_change=overall['counterfactual_change'],
             paired_both_correct=overall['paired_both_correct'],
             exact_512=r512['exact_accuracy'],nll_512=r512['target_nll'],
             train_tokens_per_sec=speed,peak_gpu_gib=gpu_gib,
             parameters=sum(p.numel() for p in model.parameters()),
             cases=overall['cases'],amp_overflows=task['overflows'],fp32_fallbacks=task['fp32_fallbacks'])
    csv_path=folder/'metrics.csv'
    # Atomic/idempotent metrics rewrite: interrupted evaluation can be safely rerun.
    previous=[]
    if csv_path.exists():
        with csv_path.open(newline='') as f:
            previous=[d for d in csv.DictReader(f) if int(d['step'])!=step]
    tmpmetrics=csv_path.with_suffix('.csv.tmp')
    with tmpmetrics.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
        writer.writerows(previous);writer.writerow(row)
    os.replace(tmpmetrics,csv_path)
    atomic_json(folder/f'gate_stats_epoch{epoch}.json',model.all_stats())
    atomic_json(folder/f'eval_summary_epoch{epoch}.json',overall)
    print(f'EPOCH_RESULT {task["variant"]} epoch={epoch} step={step} '
          f'exact={overall["exact_accuracy"]:.2%} nll={overall["target_nll"]:.4f} '
          f'ppl={overall["answer_ppl"]:.3f} exact512={r512["exact_accuracy"]:.2%}',flush=True)
    task.update(loss_sum=0.,loss_count=0,epoch_seconds=0.,last_evaluated=epoch)
    save_checkpoint(task)


def train(args):
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('Choose Kaggle T4 x2 in settings')
    torch.set_num_threads(2);seed_everything(args.seed)
    device=torch.device('cuda:0' if args.device=='cuda' else 'cpu')
    task=make_task(args,args.variant,device)
    total=args.epochs*args.steps_per_epoch
    while True:
        step=task['step']
        if step and step%args.steps_per_epoch==0:
            epoch=step//args.steps_per_epoch
            if task['last_evaluated']<epoch:
                save_checkpoint(task)  # interruption-safe before expensive evaluation
                epoch_evaluation(args,task)
        if step>=total:break
        if args.stop_after and step>=args.stop_after:break
        training_update(args,task)
        step=task['step']
        if step%args.save_every==0:
            save_checkpoint(task)
    save_checkpoint(task)
    print('DONE',task['variant'],task['step'],'of',total,flush=True)


@torch.inference_mode()
def benchmark(args):
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    device=torch.device('cuda:0' if args.device=='cuda' else 'cpu')
    task=make_task(args,args.variant,device)
    model=task['model'].eval()
    if task['step']<args.epochs*args.steps_per_epoch:
        raise RuntimeError('Benchmark speed only after 60K steps')
    rows=[]
    for context in (128,512,2048,4096):
        x,_y,_,_=make_cases(1,context,args.seed+1000+context)
        x=x.to(device)
        def prefill():return run_logits(model,x,device,128)
        prefill()
        if device.type=='cuda':torch.cuda.synchronize()
        trials=[]
        for _ in range(args.speed_repeats):
            if device.type=='cuda':
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();prefill();end.record();torch.cuda.synchronize()
                trials.append(start.elapsed_time(end))
            else:
                start=time.perf_counter();prefill();trials.append((time.perf_counter()-start)*1000)
        # Decoding with an actual cache from the full context.
        states=None
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=(device.type=='cuda')):
            for chunk in x.split(128,dim=1): _,states=model(chunk,states)
            def decode():
                nonlocal states
                _,states=model(x[:,-1:].contiguous(),states)
            decode()
            decode_times=[]
            for _ in range(args.speed_repeats):
                if device.type=='cuda':
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record();decode();end.record();torch.cuda.synchronize()
                    decode_times.append(start.elapsed_time(end))
                else:
                    start=time.perf_counter();decode();decode_times.append((time.perf_counter()-start)*1000)
        rows.append(dict(model=args.variant,context=context,
                         prefill_ms=float(np.median(trials)),
                         decode_ms=float(np.median(decode_times)),
                         prefill_tokens_per_sec=context/(float(np.median(trials))/1000),
                         decode_tokens_per_sec=1000/float(np.median(decode_times))))
    fn=task['folder']/'speed.csv'
    with fn.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for r in rows:print('SPEED',r,flush=True)


def smoke(args):
    torch.set_num_threads(2)
    # Stress-test small retention, for which cumprod/division overflows in the
    # original implementation. Verify against a direct sequential recurrence.
    generator=torch.Generator(device='cpu').manual_seed(1206)
    test_a=(.0001+.02*torch.rand(2,37,2,4,generator=generator)).requires_grad_()
    test_v=torch.randn(2,37,2,4,generator=generator,requires_grad=True)
    out,last=chunk_scan(test_a,test_v,chunk_size=13)
    st=torch.zeros_like(last);ref=[]
    for j in range(test_a.shape[1]):
        st=test_a[:,j]*st+test_v[:,j]
        ref.append(st)
    reference=torch.stack(ref,dim=1)
    assert torch.allclose(out,reference,rtol=1e-5,atol=1e-5)
    assert torch.allclose(last,reference[:,-1],rtol=1e-5,atol=1e-5)
    out.square().mean().backward()
    assert bool(torch.isfinite(test_a.grad).all()) and bool(torch.isfinite(test_v.grad).all())
    probe=nn.Parameter(torch.zeros(2))
    probe.grad=torch.full_like(probe,1.e25)
    raw_norm=stable_clip_grad_norm_([probe],1.0)
    assert math.isfinite(raw_norm) and abs(float(probe.grad.norm())-1.)<1e-4
    print('PASS division-free scan under near-zero retention + FP64 clipping',flush=True)
    a=build_model(args,'original').eval();b=build_model(args,'hybrid').eval()
    assert sum(p.numel() for p in a.parameters())<sum(p.numel() for p in b.parameters())
    for x,y in zip(a.blocks,b.blocks):
        assert torch.equal(x.esa.W_A.weight,y.esa.qgv.weight[args.dim:2*args.dim]);
        assert torch.equal(x.esa.W_V.weight,y.esa.qgv.weight[2*args.dim:]);
        assert torch.equal(x.esa.W_O.weight,y.esa.W_O.weight)
        assert torch.equal(x.esa.log_half_life,y.esa.log_half_life)
        assert torch.equal(x.esa.raw_decay_bias,y.esa.raw_decay_bias)
    print('PASS paired identical original-vs-hybrid gate/value/out weights',flush=True)
    xx,yy,alt,yy2=make_cases(2,96,772,with_counterfactual=True)
    assert (xx!=alt).sum().item()==2 and bool((yy!=yy2).all())
    for name,model in [('original',a),('hybrid',b)]:
        model.train();model.zero_grad(set_to_none=True)
        logits,_=model(xx)
        loss=F.cross_entropy(logits.float(),yy)
        loss.backward()
        bad=[n for n,p in model.named_parameters() if p.grad is None or not bool(torch.isfinite(p.grad).all())]
        assert not bad,(name,bad)
        model.eval()
        with torch.no_grad():
            whole,_=model(xx)
            states=None
            for chunk in xx.split(32,dim=1): streamed,states=model(chunk,states)
            assert torch.allclose(whole,streamed,atol=5e-4,rtol=5e-4),(name,float((whole-streamed).abs().max()))
        print('PASS',name,'gradients + full-model streaming','params',sum(p.numel() for p in model.parameters()),flush=True)
    print('ALL SMOKE TESTS PASSED',flush=True)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=('train','smoke','speed'),default='train')
    p.add_argument('--variant',choices=('original','hybrid'),default='original')
    p.add_argument('--root',default='/kaggle/working/esa_lm_original_vs_hybrid_60k')
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--steps-per-epoch',type=int,default=20000)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--dim',type=int,default=384)
    p.add_argument('--heads',type=int,default=6)
    p.add_argument('--layers',type=int,default=6)
    p.add_argument('--vocab-size',type=int,default=256)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--chunk-size',type=int,default=32)
    p.add_argument('--eval-trials',type=int,default=12,help='held-out cases per position/length at 20K and 40K')
    p.add_argument('--final-eval-trials',type=int,default=170,help='510 per length at 60K')
    p.add_argument('--eval-batch',type=int,default=3)
    p.add_argument('--save-every',type=int,default=500)
    p.add_argument('--log-every',type=int,default=200)
    p.add_argument('--stop-after',type=int,default=0)
    p.add_argument('--speed-repeats',type=int,default=5)
    p.add_argument('--resume',action=argparse.BooleanOptionalAction,default=True)
    args=p.parse_args()
    if args.heads!=6:raise ValueError('This replication uses exactly six heads')
    if args.dim%args.heads:raise ValueError('dim must be divisible by heads')
    if args.vocab_size<256:raise ValueError('vocab size must be >=256')
    return args

if __name__=='__main__':
    args=parse_args()
    if args.action=='smoke':smoke(args)
    elif args.action=='speed':benchmark(args)
    else:train(args)

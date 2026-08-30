# MLBricks Public Lifecycle API

All model lifecycle operations are architecture-agnostic and are exposed from the `mlbricks` package root.

## Save and load

```python
import mlbricks as mlb

mlb.save(model, "model")
model = mlb.load("model", device="auto")
metadata = mlb.inspect("model")
```

## Train

```python
trainer = mlb.Trainer(model, optimizer="adamw", lr=3e-4)
trainer.fit(train_loader, steps=10_000)
```

One-call form:

```python
trainer = mlb.train(model, train_loader, steps=10_000, optimizer="adamw", lr=3e-4)
```

For `(inputs, targets)` batches, MLBricks passes targets to the model automatically. Models may return a scalar loss, `(logits, loss)`, a mapping containing `loss`, or an object exposing `.loss`. For models that only return predictions, pass `loss_fn=` to `Trainer`.

## Evaluate

```python
metrics = trainer.evaluate(val_loader)
```

## Checkpoint and resume

```python
trainer.save("last")
trainer = mlb.Trainer.resume("checkpoints/last", device="auto")
```

`resume()` restores the model, optimizer when reconstructable, step/epoch, scaler state when supplied, scheduler state when supplied, and RNG state.

## Inference

```python
y = mlb.predict(model, inputs)
out = mlb.generate(model, prompt, max_new_tokens=128)
```

## Optimization

```python
model = mlb.compile(model)
model = mlb.quantize(model, method="elasticbit", bits=4)
```

## Removed ESA-specific lifecycle

The lifecycle is no longer attached to ESA. These old paths are removed:

```text
ESAModel.save(...)
ESAModel.load(...)
mlbricks.esa.Trainer
mlbricks.esa.trainer
```

`ESAModel` remains available as an ESA language-model architecture.

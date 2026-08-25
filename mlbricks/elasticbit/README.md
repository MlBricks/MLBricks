# ElasticBit

This folder contains the ElasticBit portable weight-quantization component.

- `core.py` — packing, dequantization, ElasticLinear/ElasticEmbedding, manifests
- `LICENSE_ELASTICBIT.txt` — component license notice

ElasticBit's packed CUDA linear currently uses the shared `mlbricks._C` native
extension in `mlbricks/bolt`, which is also used by ESA. The Python component is
now isolated here while keeping `from mlbricks.elasticbit import ...`
backward compatible.

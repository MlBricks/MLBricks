"""ElasticBit threshold-driven compression example."""

import torch
from mlbricks import ElasticBit

# Move the model to CUDA FP16 before compression.
model = torch.nn.Sequential(
    torch.nn.Linear(128, 256),
    torch.nn.GELU(),
    torch.nn.Linear(256, 64),
).cuda().half().eval()

# Representative model inputs. ElasticBit captures each Linear's activation
# rows and chooses the smallest safe storage width for that matrix.
calibrationData = [torch.randn(4, 128, device="cuda", dtype=torch.float16)]

model = ElasticBit.compress(
    model,
    calibrationData=calibrationData,
    threshold=0.01,
)

model.elasticbit.summary()

# Decode policy can change without recalibration/recompression.
model.elasticbit.setDecodePolicy("fullPrecision")
model.elasticbit.setDecodePolicy("hardwareNative")

ElasticBit.save(model, "model.elasticbit")

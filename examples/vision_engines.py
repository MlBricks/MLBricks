import torch

from mlbricks import Vesa, VisionBolt

# Classification: scan-derived spatial order, no position embedding.
vesa = Vesa(
    image_size=32, patch_size=4, num_classes=10, dim=192, depth=6,
    engine="Serpentine", position=None, scan="cross",
)

bolt = VisionBolt(
    image_size=32, patch_size=4, num_classes=10, dim=192, depth=6,
    heads=6, latent_dim=32,
    engine="Serpentine", position=None, scan="cross",
)

images = torch.randn(2, 3, 32, 32)
print(vesa(images).shape, bolt(images).shape)

# ViT style: auto position becomes 2-D sin/cos.
vit_bolt = VisionBolt(
    image_size=32, patch_size=4, num_classes=10, dim=192, depth=6,
    heads=6, latent_dim=32, engine="ViT",
)
print(vit_bolt(images).shape)

# Diffusion: input/output are images.
diff = VisionBolt(
    image_size=32, patch_size=4, dim=192, depth=6, heads=6, latent_dim=32,
    engine="Diffusion", position=None, scan="cross",
)
print(diff(images, torch.tensor([10, 20])).shape)

# AR: visual token IDs.
ar = Vesa(
    image_size=32, patch_size=4, vocab_size=8192, dim=192, depth=6,
    engine="AR", position=None, scan="cross",
)
ids = torch.randint(0, 8192, (2, 16))
print(ar(ids).shape, ar.generate(ids, 4).shape)

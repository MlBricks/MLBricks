from mlbricks import Bolt

layer = Bolt(384, 6, latent_dim=32)  # backend="auto" implicitly
layer.set_backend("pytorch")
layer.set_backend("native")
layer.set_backend("auto")

from mlbricks import Vesa


model = Vesa(
    image_size=32,
    patch_size=4,
    in_channels=3,
    num_classes=10,
    dim=192,
    depth=4,
)

# backend="auto" is the default; the planner resolves available routes.
print(model)
print("Backend:", model.config.backend)

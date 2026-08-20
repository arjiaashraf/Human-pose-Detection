import torch

from src.softargmax import SoftArgmax2D


print("=" * 60)
print("Testing SoftArgmax2D")
print("=" * 60)


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ------------------------------------------------------------
# Create layer
# ------------------------------------------------------------

softargmax = SoftArgmax2D(
    beta=100.0
).to(device)

print("\nSoftArgmax2D created successfully!")


# ------------------------------------------------------------
# Create test heatmaps
# ------------------------------------------------------------

heatmaps = torch.randn(
    2,
    17,
    64,
    64,
    device=device,
    requires_grad=True
)

print("\nInput heatmap shape:")
print(heatmaps.shape)


# ------------------------------------------------------------
# Forward pass
# ------------------------------------------------------------

coordinates = softargmax(
    heatmaps
)

print("\nOutput coordinate shape:")
print(coordinates.shape)


# ------------------------------------------------------------
# Expected shape
# ------------------------------------------------------------

expected_shape = (
    2,
    17,
    2
)

print("\nExpected shape:")
print(expected_shape)


assert coordinates.shape == expected_shape

print("\n✓ Coordinate shape is correct!")


# ------------------------------------------------------------
# Check coordinate range
# ------------------------------------------------------------

print("\nCoordinate minimum:")
print(coordinates.min().item())

print("\nCoordinate maximum:")
print(coordinates.max().item())


assert coordinates.min() >= 0
assert coordinates.max() <= 63

print("\n✓ Coordinates are within 64 × 64 heatmap!")


# ------------------------------------------------------------
# Print example
# ------------------------------------------------------------

print("\nExample coordinates:")

print(
    coordinates[0, :5]
)


# ------------------------------------------------------------
# Gradient test
# ------------------------------------------------------------

loss = coordinates.mean()

loss.backward()

print("\n✓ Backward pass successful!")

print("\nGradient exists:")

print(
    heatmaps.grad is not None
)

assert heatmaps.grad is not None

print("\n✓ Gradients successfully flow through SoftArgmax2D!")


# ------------------------------------------------------------
# Final
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("SOFTARGMAX2D TEST PASSED!")
print("=" * 60)
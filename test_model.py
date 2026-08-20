import torch
from src.model import UNetPose


print("=" * 60)
print("Testing Multi-Task UNetPose")
print("=" * 60)


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model = UNetPose(
    in_channels=3,
    num_keypoints=17
).to(device)

print("\nModel created successfully!")


# ------------------------------------------------------------
# Test input
# ------------------------------------------------------------

x = torch.randn(
    2,
    3,
    256,
    256
).to(device)

print("\nInput shape:")
print(x.shape)


# ------------------------------------------------------------
# Forward pass
# ------------------------------------------------------------

heatmaps, visibility = model(x)


print("\nHeatmap output:")
print(heatmaps.shape)

print("\nVisibility output:")
print(visibility.shape)


# ------------------------------------------------------------
# Expected shapes
# ------------------------------------------------------------

expected_heatmap = (
    2,
    17,
    64,
    64
)

expected_visibility = (
    2,
    17
)

print("\nExpected heatmap shape:")
print(expected_heatmap)

print("\nExpected visibility shape:")
print(expected_visibility)


# ------------------------------------------------------------
# Verify shapes
# ------------------------------------------------------------

assert heatmaps.shape == expected_heatmap

assert visibility.shape == expected_visibility

print("\n✓ Heatmap shape is correct!")

print("✓ Visibility shape is correct!")


# ------------------------------------------------------------
# Test losses
# ------------------------------------------------------------

target_heatmaps = torch.rand_like(
    heatmaps
)

target_visibility = torch.randint(
    0,
    2,
    visibility.shape
).float()


mse_loss = torch.nn.functional.mse_loss(
    heatmaps,
    target_heatmaps
)

bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
    visibility,
    target_visibility
)

total_loss = mse_loss + bce_loss


print("\nMSE loss:")
print(mse_loss.item())

print("\nBCE loss:")
print(bce_loss.item())

print("\nTotal loss:")
print(total_loss.item())


# ------------------------------------------------------------
# Backward test
# ------------------------------------------------------------

total_loss.backward()

print("\n✓ MSE + BCE backward pass successful!")


# ------------------------------------------------------------
# Final
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MULTI-TASK MODEL TEST PASSED!")
print("=" * 60)
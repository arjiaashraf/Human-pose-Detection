import torch
from src.softargmax import SoftArgmax2D


def test_softargmax():
    print("=" * 60)
    print("Testing SoftArgmax2D")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    softargmax = SoftArgmax2D(beta=100.0).to(device)
    print("\nSoftArgmax2D created successfully!")

    # 1. Test standard shapes
    heatmaps = torch.randn(2, 17, 64, 64, device=device, requires_grad=True)
    coordinates = softargmax(heatmaps)

    expected_shape = (2, 17, 2)
    assert coordinates.shape == expected_shape, f"Expected {expected_shape}, got {coordinates.shape}"
    print(f"\n✓ Coordinate shape is correct: {coordinates.shape}")

    # 2. Check coordinate boundaries
    assert coordinates.min() >= 0, "Coordinates contain negative values!"
    assert coordinates.max() <= 63, "Coordinates exceed maximum heatmap size!"
    print("\n✓ Coordinates are strictly within standard 64 × 64 heatmap bounds!")

    # 3. Peak Detection Accuracy Test
    test_peak = torch.zeros(1, 1, 64, 64, device=device)
    target_y, target_x = 20, 10
    test_peak[0, 0, target_y, target_x] = 10.0  # Strong peak activation

    predicted_coord = softargmax(test_peak)
    pred_x, pred_y = predicted_coord[0, 0, 0].item(), predicted_coord[0, 0, 1].item()

    print(f"\nSynthetic Peak Location Test:")
    print(f"Target: (x={target_x}, y={target_y})")
    print(f"Predicted: (x={pred_x:.2f}, y={pred_y:.2f})")

    assert abs(pred_x - target_x) < 0.1 and abs(pred_y - target_y) < 0.1, "SoftArgmax failed to locate heatmap peak!"
    print("✓ Peak detection test passed accurately!")

    # 4. Gradient Flow Test
    loss = coordinates.mean()
    loss.backward()
    assert heatmaps.grad is not None, "Gradients failed to flow back into heatmaps!"
    print("\n✓ Gradients successfully flow through SoftArgmax2D!")

    print("\n" + "=" * 60)
    print("SOFTARGMAX2D ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_softargmax()
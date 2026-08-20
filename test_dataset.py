import sys
import os
import torch

# Make sure Python can find src/dataset.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dataset import COCOKeypointDataset


# --------------------------------------------------
# Paths
# --------------------------------------------------

IMAGE_DIR = r"C:\Users\Computer House\fiftyone\coco-2017\train\data"

ANNOTATION_FILE = (
    r"C:\Users\Computer House\fiftyone\coco-2017\raw\person_keypoints_train2017.json"
)


# --------------------------------------------------
# Create Dataset
# --------------------------------------------------

print("=" * 60)
print("Creating COCO Keypoint Dataset...")
print("=" * 60)

dataset = COCOKeypointDataset(
    image_dir=IMAGE_DIR,
    annotation_file=ANNOTATION_FILE,
    image_size=256,
    heatmap_size=64,
    sigma=2.0
)


# --------------------------------------------------
# Dataset information
# --------------------------------------------------

print("\nDataset created successfully!")
print("Dataset size:", len(dataset))


# --------------------------------------------------
# Test first sample
# --------------------------------------------------

print("\nLoading first sample...")

image, heatmaps, visibility = dataset[0]


# --------------------------------------------------
# Print shapes
# --------------------------------------------------

print("\n" + "=" * 60)
print("SAMPLE INFORMATION")
print("=" * 60)

print("Image shape:")
print(image.shape)

print("\nHeatmaps shape:")
print(heatmaps.shape)

print("\nVisibility shape:")
print(visibility.shape)


# --------------------------------------------------
# Check values
# --------------------------------------------------

print("\n" + "=" * 60)
print("VALUE CHECK")
print("=" * 60)

print("Image dtype:", image.dtype)
print("Heatmaps dtype:", heatmaps.dtype)
print("Visibility dtype:", visibility.dtype)

print("\nImage min:", image.min().item())
print("Image max:", image.max().item())

print("\nHeatmap min:", heatmaps.min().item())
print("Heatmap max:", heatmaps.max().item())

print("\nVisibility:")
print(visibility)


# --------------------------------------------------
# Check individual keypoints
# --------------------------------------------------

print("\n" + "=" * 60)
print("KEYPOINT HEATMAP CHECK")
print("=" * 60)

for i in range(17):
    maximum = heatmaps[i].max().item()

    print(
        f"Keypoint {i:2d}: "
        f"visible={int(visibility[i].item())}, "
        f"max_heatmap={maximum:.4f}"
    )


print("\n" + "=" * 60)
print("DATASET TEST COMPLETED SUCCESSFULLY!")
print("=" * 60)
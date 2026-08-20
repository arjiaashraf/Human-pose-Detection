import os
import cv2
import torch
import numpy as np

from src.model import UNetPose
from src.softargmax import SoftArgmax2D


# ============================================================
# Configuration
# ============================================================

IMAGE_PATH = (
    r"C:\Users\Computer House\fiftyone\coco-2017"
    r"\train\data\000000000036.jpg"
)

MODEL_PATH = (
    r"checkpoints\unet_pose_test.pth"
)

OUTPUT_PATH = (
    r"pose_result.jpg"
)

IMAGE_SIZE = 256
HEATMAP_SIZE = 64
NUM_KEYPOINTS = 17


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# COCO Skeleton
# ============================================================

SKELETON = [
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
]


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("TESTING POSE INFERENCE")
    print("=" * 60)

    print("\nDevice:", DEVICE)


    # ========================================================
    # Check files
    # ========================================================

    if not os.path.exists(IMAGE_PATH):

        raise FileNotFoundError(
            f"Image not found:\n{IMAGE_PATH}"
        )

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )


    # ========================================================
    # Load Model
    # ========================================================

    print("\nLoading UNetPose...")

    model = UNetPose(
        in_channels=3,
        num_keypoints=NUM_KEYPOINTS
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    model = model.to(DEVICE)

    model.eval()

    print("Model loaded successfully!")


    # ========================================================
    # Load SoftArgmax
    # ========================================================

    softargmax = SoftArgmax2D()

    print(
        "SoftArgmax2D loaded successfully!"
    )


    # ========================================================
    # Load Image
    # ========================================================

    print("\nLoading image...")

    original_image = cv2.imread(
        IMAGE_PATH
    )

    if original_image is None:

        raise RuntimeError(
            f"Could not load image:\n{IMAGE_PATH}"
        )

    original_h, original_w = (
        original_image.shape[:2]
    )

    print(
        "Original image size:",
        original_w,
        "x",
        original_h
    )


    # ========================================================
    # Prepare Image
    # ========================================================

    image_rgb = cv2.cvtColor(
        original_image,
        cv2.COLOR_BGR2RGB
    )

    resized = cv2.resize(
        image_rgb,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    resized = (
        resized.astype(np.float32)
        / 255.0
    )

    resized = np.transpose(
        resized,
        (2, 0, 1)
    )

    tensor = torch.tensor(
        resized,
        dtype=torch.float32
    )

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)


    # ========================================================
    # Model Inference
    # ========================================================

    print("\nRunning model inference...")

    with torch.no_grad():

        heatmaps, visibility = model(
            tensor
        )


    print(
        "Heatmap shape:",
        heatmaps.shape
    )

    print(
        "Visibility shape:",
        visibility.shape
    )


    # ========================================================
    # SoftArgmax
    # ========================================================

    print("\nConverting heatmaps to coordinates...")

    with torch.no_grad():

        coordinates = softargmax(
            heatmaps
        )

        visibility_probability = (
            torch.sigmoid(visibility)
        )


    coordinates = (
        coordinates[0]
        .cpu()
        .numpy()
    )

    visibility_probability = (
        visibility_probability[0]
        .cpu()
        .numpy()
    )


    # ========================================================
    # Convert 64x64 coordinates
    # to original image coordinates
    # ========================================================

    scale_x = (
        original_w
        / HEATMAP_SIZE
    )

    scale_y = (
        original_h
        / HEATMAP_SIZE
    )


    coordinates[:, 0] *= scale_x
    coordinates[:, 1] *= scale_y


    # ========================================================
    # Print Predictions
    # ========================================================

    print("\nPredicted keypoints:")
    print("-" * 60)

    keypoint_names = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]


    for i in range(NUM_KEYPOINTS):

        x = coordinates[i][0]
        y = coordinates[i][1]

        confidence = (
            visibility_probability[i]
        )

        print(
            f"{i:2d} "
            f"{keypoint_names[i]:15s} "
            f"x={x:7.2f} "
            f"y={y:7.2f} "
            f"visibility={confidence:.3f}"
        )


    # ========================================================
    # Draw Keypoints
    # ========================================================

    result = original_image.copy()

    CONFIDENCE_THRESHOLD = 0.3


    for i in range(NUM_KEYPOINTS):

        x = int(
            coordinates[i][0]
        )

        y = int(
            coordinates[i][1]
        )

        confidence = (
            visibility_probability[i]
        )

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        if (
            x < 0
            or x >= original_w
            or y < 0
            or y >= original_h
        ):
            continue

        cv2.circle(
            result,
            (x, y),
            5,
            (0, 255, 0),
            -1
        )


    # ========================================================
    # Draw Skeleton
    # ========================================================

    for joint_a, joint_b in SKELETON:

        confidence_a = (
            visibility_probability[joint_a]
        )

        confidence_b = (
            visibility_probability[joint_b]
        )

        if (
            confidence_a < CONFIDENCE_THRESHOLD
            or
            confidence_b < CONFIDENCE_THRESHOLD
        ):
            continue


        x1 = int(
            coordinates[joint_a][0]
        )

        y1 = int(
            coordinates[joint_a][1]
        )

        x2 = int(
            coordinates[joint_b][0]
        )

        y2 = int(
            coordinates[joint_b][1]
        )


        cv2.line(
            result,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )


    # ========================================================
    # Save Result
    # ========================================================

    cv2.imwrite(
        OUTPUT_PATH,
        result
    )


    print("\nPose visualization saved to:")

    print(
        os.path.abspath(
            OUTPUT_PATH
        )
    )


    print("\n" + "=" * 60)
    print("POSE INFERENCE TEST COMPLETED!")
    print("=" * 60)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()
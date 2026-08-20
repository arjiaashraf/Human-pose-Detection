import os

import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    random_split,
    Subset
)

from src.dataset import COCOKeypointDataset
from src.model import UNetPose


# ============================================================
# Configuration
# ============================================================

IMAGE_DIR = r"C:\Users\Computer House\fiftyone\coco-2017\train\data"

ANNOTATION_FILE = (
    r"C:\Users\Computer House\fiftyone\coco-2017\raw"
    r"\person_keypoints_train2017.json"
)

# CPU testing
BATCH_SIZE = 2
EPOCHS = 1

LEARNING_RATE = 1e-3

IMAGE_SIZE = 256
HEATMAP_SIZE = 64

TRAIN_RATIO = 0.8

# ------------------------------------------------------------
# IMPORTANT
#
# Start with only 500 samples because you are using CPU.
#
# Later:
# MAX_SAMPLES = None
#
# ------------------------------------------------------------

MAX_SAMPLES = 500


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("COCO HUMAN POSE TRAINING")
    print("=" * 60)

    print(f"\nDevice: {DEVICE}")

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    else:

        print(
            "WARNING: Training on CPU."
        )


    # ========================================================
    # Dataset
    # ========================================================

    print("\nCreating dataset...")

    dataset = COCOKeypointDataset(

        image_dir=IMAGE_DIR,

        annotation_file=ANNOTATION_FILE,

        image_size=IMAGE_SIZE,

        heatmap_size=HEATMAP_SIZE,

        sigma=2.0
    )

    print(
        "Full dataset size:",
        len(dataset)
    )


    # ========================================================
    # CPU Subset
    # ========================================================

    if MAX_SAMPLES is not None:

        max_samples = min(
            MAX_SAMPLES,
            len(dataset)
        )

        dataset = Subset(
            dataset,
            range(max_samples)
        )

        print(
            "Using subset for CPU testing:",
            len(dataset)
        )


    # ========================================================
    # Train / Validation Split
    # ========================================================

    train_size = int(
        TRAIN_RATIO * len(dataset)
    )

    val_size = (
        len(dataset)
        - train_size
    )

    train_dataset, val_dataset = random_split(

        dataset,

        [
            train_size,
            val_size
        ]
    )

    print(
        "Training samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )


    # ========================================================
    # DataLoaders
    # ========================================================

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0
    )

    print(
        "Training batches:",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(val_loader)
    )


    # ========================================================
    # Model
    # ========================================================

    print("\nCreating model...")

    model = UNetPose(

        in_channels=3,

        num_keypoints=17

    ).to(DEVICE)

    print(
        "Model created successfully."
    )


    # ========================================================
    # Loss Functions
    # ========================================================

    heatmap_loss_fn = nn.MSELoss()

    visibility_loss_fn = (
        nn.BCEWithLogitsLoss()
    )


    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=LEARNING_RATE
    )


    # ========================================================
    # Training
    # ========================================================

    print("\nStarting training...")

    print("=" * 60)


    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        running_heatmap_loss = 0.0

        running_visibility_loss = 0.0


        # ====================================================
        # Training Batches
        # ====================================================

        for batch_idx, (
            images,
            target_heatmaps,
            visibility
        ) in enumerate(train_loader):


            # ------------------------------------------------
            # Move data to device
            # ------------------------------------------------

            images = images.to(DEVICE)

            target_heatmaps = (
                target_heatmaps.to(DEVICE)
            )

            visibility = (
                visibility.to(DEVICE)
            )


            # ------------------------------------------------
            # Forward Pass
            # ------------------------------------------------

            heatmap_output, visibility_output = (
                model(images)
            )


            # ------------------------------------------------
            # Shape Check
            # ------------------------------------------------

            if heatmap_output.shape != (
                images.shape[0],
                17,
                HEATMAP_SIZE,
                HEATMAP_SIZE
            ):

                raise RuntimeError(
                    "Incorrect heatmap output shape: "
                    f"{heatmap_output.shape}"
                )


            if visibility_output.shape != (
                images.shape[0],
                17
            ):

                raise RuntimeError(
                    "Incorrect visibility output shape: "
                    f"{visibility_output.shape}"
                )


            # ------------------------------------------------
            # Heatmap MSE Loss
            # ------------------------------------------------

            heatmap_loss = heatmap_loss_fn(

                heatmap_output,

                target_heatmaps
            )


            # ------------------------------------------------
            # Visibility BCE Loss
            # ------------------------------------------------

            visibility_loss = (
                visibility_loss_fn(

                    visibility_output,

                    visibility
                )
            )


            # ------------------------------------------------
            # Combined Multi-Task Loss
            # ------------------------------------------------

            total_loss = (

                heatmap_loss

                + visibility_loss
            )


            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            optimizer.zero_grad()

            total_loss.backward()

            optimizer.step()


            # ------------------------------------------------
            # Accumulate Losses
            # ------------------------------------------------

            running_loss += (
                total_loss.item()
            )

            running_heatmap_loss += (
                heatmap_loss.item()
            )

            running_visibility_loss += (
                visibility_loss.item()
            )


            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (batch_idx + 1) % 25 == 0:

                print(

                    f"Epoch [{epoch + 1}/{EPOCHS}] "

                    f"Batch [{batch_idx + 1}/"
                    f"{len(train_loader)}] "

                    f"Total Loss: "
                    f"{total_loss.item():.4f} | "

                    f"Heatmap: "
                    f"{heatmap_loss.item():.4f} | "

                    f"Visibility: "
                    f"{visibility_loss.item():.4f}"

                )


        # ====================================================
        # Average Training Loss
        # ====================================================

        num_batches = len(train_loader)

        average_loss = (

            running_loss

            / num_batches
        )

        average_heatmap_loss = (

            running_heatmap_loss

            / num_batches
        )

        average_visibility_loss = (

            running_visibility_loss

            / num_batches
        )


        print("\n" + "-" * 60)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} completed"
        )

        print(
            f"Average Total Loss: "
            f"{average_loss:.4f}"
        )

        print(
            f"Average Heatmap MSE: "
            f"{average_heatmap_loss:.4f}"
        )

        print(
            f"Average Visibility BCE: "
            f"{average_visibility_loss:.4f}"
        )

        print("-" * 60)


    # ========================================================
    # Save Model
    # ========================================================

    os.makedirs(
        "checkpoints",
        exist_ok=True
    )

    model_path = (
        "checkpoints/"
        "unet_pose_test.pth"
    )

    torch.save(

        model.state_dict(),

        model_path
    )

    print("\nModel saved to:")

    print(
        os.path.abspath(model_path)
    )


    # ========================================================
    # Finished
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "TRAINING PIPELINE TEST COMPLETED!"
    )

    print("=" * 60)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()
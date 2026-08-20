import os
import random

import numpy as np
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

IMAGE_DIR = (
    r"C:\Users\Computer House\fiftyone\coco-2017\train\data"
)

ANNOTATION_FILE = (
    r"C:\Users\Computer House\fiftyone\coco-2017\raw"
    r"\person_keypoints_train2017.json"
)

# ------------------------------------------------------------
# Experiment 2
# ------------------------------------------------------------

MAX_SAMPLES = 2000

BATCH_SIZE = 4

EPOCHS = 5

LEARNING_RATE = 1e-3

IMAGE_SIZE = 256

HEATMAP_SIZE = 64

TRAIN_RATIO = 0.8

SEED = 42


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Main
# ============================================================

def main():

    set_seed(SEED)

    print("=" * 70)
    print("COCO HUMAN POSE TRAINING - EXPERIMENT 2")
    print("=" * 70)

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

    print("\nConfiguration:")
    print(
        f"Maximum samples: {MAX_SAMPLES}"
    )
    print(
        f"Batch size: {BATCH_SIZE}"
    )
    print(
        f"Epochs: {EPOCHS}"
    )
    print(
        f"Learning rate: {LEARNING_RATE}"
    )
    print(
        f"Random seed: {SEED}"
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
    # Subset
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
            "Using subset:",
            len(dataset)
        )

    # ========================================================
    # Train / Validation split
    # ========================================================

    train_size = int(
        TRAIN_RATIO * len(dataset)
    )

    val_size = (
        len(dataset)
        - train_size
    )

    generator = torch.Generator()

    generator.manual_seed(SEED)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
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
    # Loss functions
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
    # Checkpoint directory
    # ========================================================

    checkpoint_dir = "checkpoints"

    os.makedirs(
        checkpoint_dir,
        exist_ok=True
    )

    best_model_path = os.path.join(
        checkpoint_dir,
        "best_unet_pose.pth"
    )

    final_model_path = os.path.join(
        checkpoint_dir,
        "unet_pose_final.pth"
    )

    # ========================================================
    # Best validation loss
    # ========================================================

    best_val_loss = float("inf")

    # ========================================================
    # Training
    # ========================================================

    print("\nStarting training...")
    print("=" * 70)

    for epoch in range(EPOCHS):

        # ====================================================
        # TRAINING
        # ====================================================

        model.train()

        running_train_loss = 0.0

        running_train_heatmap = 0.0

        running_train_visibility = 0.0

        print(
            f"\nEpoch [{epoch + 1}/{EPOCHS}]"
        )

        print("-" * 70)

        for batch_idx, (
            images,
            target_heatmaps,
            visibility
        ) in enumerate(train_loader):

            # ------------------------------------------------
            # Move to device
            # ------------------------------------------------

            images = images.to(DEVICE)

            target_heatmaps = (
                target_heatmaps.to(DEVICE)
            )

            visibility = (
                visibility.to(DEVICE)
            )

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            heatmap_output, visibility_output = (
                model(images)
            )

            # ------------------------------------------------
            # Shape checks
            # ------------------------------------------------

            expected_heatmap_shape = (
                images.shape[0],
                17,
                HEATMAP_SIZE,
                HEATMAP_SIZE
            )

            expected_visibility_shape = (
                images.shape[0],
                17
            )

            if heatmap_output.shape != (
                expected_heatmap_shape
            ):

                raise RuntimeError(
                    "Incorrect heatmap output shape: "
                    f"{heatmap_output.shape}"
                )

            if visibility_output.shape != (
                expected_visibility_shape
            ):

                raise RuntimeError(
                    "Incorrect visibility output shape: "
                    f"{visibility_output.shape}"
                )

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            heatmap_loss = heatmap_loss_fn(
                heatmap_output,
                target_heatmaps
            )

            visibility_loss = visibility_loss_fn(
                visibility_output,
                visibility
            )

            total_loss = (
                heatmap_loss
                +
                visibility_loss
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            optimizer.zero_grad()

            total_loss.backward()

            optimizer.step()

            # ------------------------------------------------
            # Accumulate
            # ------------------------------------------------

            running_train_loss += (
                total_loss.item()
            )

            running_train_heatmap += (
                heatmap_loss.item()
            )

            running_train_visibility += (
                visibility_loss.item()
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                batch_idx + 1
            ) % 50 == 0:

                print(
                    f"Train Batch "
                    f"[{batch_idx + 1}/"
                    f"{len(train_loader)}] "
                    f"Loss: "
                    f"{total_loss.item():.4f} | "
                    f"Heatmap: "
                    f"{heatmap_loss.item():.4f} | "
                    f"Visibility: "
                    f"{visibility_loss.item():.4f}"
                )

        # ====================================================
        # Average training losses
        # ====================================================

        train_batches = len(train_loader)

        avg_train_loss = (
            running_train_loss
            /
            train_batches
        )

        avg_train_heatmap = (
            running_train_heatmap
            /
            train_batches
        )

        avg_train_visibility = (
            running_train_visibility
            /
            train_batches
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        running_val_loss = 0.0

        running_val_heatmap = 0.0

        running_val_visibility = 0.0

        with torch.no_grad():

            for (
                images,
                target_heatmaps,
                visibility
            ) in val_loader:

                images = images.to(DEVICE)

                target_heatmaps = (
                    target_heatmaps.to(DEVICE)
                )

                visibility = (
                    visibility.to(DEVICE)
                )

                # --------------------------------------------
                # Forward
                # --------------------------------------------

                heatmap_output, visibility_output = (
                    model(images)
                )

                # --------------------------------------------
                # Validation loss
                # --------------------------------------------

                heatmap_loss = heatmap_loss_fn(
                    heatmap_output,
                    target_heatmaps
                )

                visibility_loss = visibility_loss_fn(
                    visibility_output,
                    visibility
                )

                total_loss = (
                    heatmap_loss
                    +
                    visibility_loss
                )

                # --------------------------------------------
                # Accumulate
                # --------------------------------------------

                running_val_loss += (
                    total_loss.item()
                )

                running_val_heatmap += (
                    heatmap_loss.item()
                )

                running_val_visibility += (
                    visibility_loss.item()
                )

        # ====================================================
        # Average validation losses
        # ====================================================

        val_batches = len(val_loader)

        avg_val_loss = (
            running_val_loss
            /
            val_batches
        )

        avg_val_heatmap = (
            running_val_heatmap
            /
            val_batches
        )

        avg_val_visibility = (
            running_val_visibility
            /
            val_batches
        )

        # ====================================================
        # Epoch results
        # ====================================================

        print("\n" + "=" * 70)

        print(
            f"EPOCH {epoch + 1}/{EPOCHS} RESULTS"
        )

        print("=" * 70)

        print(
            f"Training Total Loss: "
            f"{avg_train_loss:.4f}"
        )

        print(
            f"Training Heatmap MSE: "
            f"{avg_train_heatmap:.4f}"
        )

        print(
            f"Training Visibility BCE: "
            f"{avg_train_visibility:.4f}"
        )

        print()

        print(
            f"Validation Total Loss: "
            f"{avg_val_loss:.4f}"
        )

        print(
            f"Validation Heatmap MSE: "
            f"{avg_val_heatmap:.4f}"
        )

        print(
            f"Validation Visibility BCE: "
            f"{avg_val_visibility:.4f}"
        )

        # ====================================================
        # Best model
        # ====================================================

        if avg_val_loss < best_val_loss:

            best_val_loss = avg_val_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "train_loss": avg_train_loss,
                },
                best_model_path
            )

            print("\n*** NEW BEST MODEL SAVED ***")

            print(
                "Path:",
                os.path.abspath(
                    best_model_path
                )
            )

        else:

            print(
                "\nNo improvement in validation loss."
            )

        print("=" * 70)

    # ========================================================
    # Save final model
    # ========================================================

    torch.save(
        {
            "epoch": EPOCHS,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val_loss,
            "train_loss": avg_train_loss,
        },
        final_model_path
    )

    # ========================================================
    # Finished
    # ========================================================

    print("\n")
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(
        "\nBest validation loss:",
        f"{best_val_loss:.4f}"
    )

    print(
        "\nBest model:"
    )

    print(
        os.path.abspath(
            best_model_path
        )
    )

    print(
        "\nFinal model:"
    )

    print(
        os.path.abspath(
            final_model_path
        )
    )

    print("\n" + "=" * 70)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()
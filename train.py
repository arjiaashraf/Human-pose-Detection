import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset

from src.dataset import COCOKeypointDataset
from src.model import UNetPose


# ============================================================
# CONFIGURATION - EXPERIMENT 3
# ============================================================

IMAGE_DIR = (
    r"C:\Users\Computer House\fiftyone\coco-2017\train\data"
)

ANNOTATION_FILE = (
    r"C:\Users\Computer House\fiftyone\coco-2017\raw"
    r"\person_keypoints_train2017.json"
)

CHECKPOINT_DIR = (
    r"checkpoints\experiment3"
)

BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_unet_pose.pth"
)

FINAL_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "unet_pose_final.pth"
)

IMAGE_SIZE = 256
HEATMAP_SIZE = 64
NUM_KEYPOINTS = 17

MAX_SAMPLES = 2000

BATCH_SIZE = 4

EPOCHS = 20

LEARNING_RATE = 0.0001

TRAIN_RATIO = 0.80

SIGMA = 2.5

# Loss weights
HEATMAP_WEIGHT = 10.0
VISIBILITY_WEIGHT = 0.5

RANDOM_SEED = 42


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# HEATMAP LOSS
# ============================================================

def heatmap_loss(
    predicted,
    target,
    visibility
):
    """
    Weighted MSE.

    Visible keypoints receive full weight.
    Invisible/unlabeled keypoints are ignored.
    """

    # [B, K, H, W]
    joint_visibility = visibility.unsqueeze(
        -1
    ).unsqueeze(
        -1
    )

    squared_error = (
        predicted - target
    ) ** 2

    # Ignore unlabeled joints
    weighted_error = (
        squared_error * joint_visibility
    )

    denominator = (
        joint_visibility.sum()
        * predicted.shape[-1]
        * predicted.shape[-2]
    )

    denominator = torch.clamp(
        denominator,
        min=1.0
    )

    return weighted_error.sum() / denominator


# ============================================================
# VISIBILITY LOSS
# ============================================================

visibility_criterion = nn.BCEWithLogitsLoss()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("COCO HUMAN POSE TRAINING - EXPERIMENT 3")
    print("=" * 70)

    print("\nDevice:", DEVICE)

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
        "Maximum samples:",
        MAX_SAMPLES
    )
    print(
        "Batch size:",
        BATCH_SIZE
    )
    print(
        "Epochs:",
        EPOCHS
    )
    print(
        "Learning rate:",
        LEARNING_RATE
    )
    print(
        "Heatmap weight:",
        HEATMAP_WEIGHT
    )
    print(
        "Visibility weight:",
        VISIBILITY_WEIGHT
    )
    print(
        "Heatmap sigma:",
        SIGMA
    )
    print(
        "Random seed:",
        RANDOM_SEED
    )

    # ========================================================
    # Checkpoint directory
    # ========================================================

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
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
        sigma=SIGMA
    )

    print(
        "Full dataset size:",
        len(dataset)
    )

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
    # Train / validation split
    # ========================================================

    train_size = int(
        TRAIN_RATIO * len(dataset)
    )

    val_size = (
        len(dataset)
        - train_size
    )

    generator = torch.Generator()

    generator.manual_seed(
        RANDOM_SEED
    )

    train_dataset, val_dataset = (
        random_split(
            dataset,
            [train_size, val_size],
            generator=generator
        )
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
    # DataLoader
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
        num_keypoints=NUM_KEYPOINTS
    )

    model = model.to(DEVICE)

    print(
        "Model created successfully."
    )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ========================================================
    # Learning rate scheduler
    # ========================================================

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    # ========================================================
    # Best validation loss
    # ========================================================

    best_val_loss = float("inf")

    # ========================================================
    # TRAINING
    # ========================================================

    print("\nStarting training...")

    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        running_total = 0.0
        running_heatmap = 0.0
        running_visibility = 0.0

        print(
            f"\nEpoch [{epoch}/{EPOCHS}]"
        )

        print("-" * 70)

        for batch_idx, (
            images,
            target_heatmaps,
            visibility
        ) in enumerate(train_loader):

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

            predicted_heatmaps, predicted_visibility = (
                model(images)
            )

            # ------------------------------------------------
            # Losses
            # ------------------------------------------------

            hm_loss = heatmap_loss(
                predicted_heatmaps,
                target_heatmaps,
                visibility
            )

            vis_loss = visibility_criterion(
                predicted_visibility,
                visibility
            )

            total_loss = (
                HEATMAP_WEIGHT * hm_loss
                +
                VISIBILITY_WEIGHT * vis_loss
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            optimizer.zero_grad()

            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            running_total += (
                total_loss.item()
            )

            running_heatmap += (
                hm_loss.item()
            )

            running_visibility += (
                vis_loss.item()
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                batch_idx + 1
            ) % 50 == 0:

                print(
                    f"Train Batch "
                    f"[{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {total_loss.item():.4f} | "
                    f"Heatmap: {hm_loss.item():.4f} | "
                    f"Visibility: {vis_loss.item():.4f}"
                )

        # ====================================================
        # Average training losses
        # ====================================================

        train_total = (
            running_total
            /
            len(train_loader)
        )

        train_heatmap = (
            running_heatmap
            /
            len(train_loader)
        )

        train_visibility = (
            running_visibility
            /
            len(train_loader)
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_total_sum = 0.0
        val_heatmap_sum = 0.0
        val_visibility_sum = 0.0

        with torch.no_grad():

            for images, target_heatmaps, visibility in val_loader:

                images = images.to(DEVICE)

                target_heatmaps = (
                    target_heatmaps.to(DEVICE)
                )

                visibility = (
                    visibility.to(DEVICE)
                )

                predicted_heatmaps, predicted_visibility = (
                    model(images)
                )

                hm_loss = heatmap_loss(
                    predicted_heatmaps,
                    target_heatmaps,
                    visibility
                )

                vis_loss = visibility_criterion(
                    predicted_visibility,
                    visibility
                )

                total_loss = (
                    HEATMAP_WEIGHT * hm_loss
                    +
                    VISIBILITY_WEIGHT * vis_loss
                )

                val_total_sum += (
                    total_loss.item()
                )

                val_heatmap_sum += (
                    hm_loss.item()
                )

                val_visibility_sum += (
                    vis_loss.item()
                )

        val_total = (
            val_total_sum
            /
            len(val_loader)
        )

        val_heatmap = (
            val_heatmap_sum
            /
            len(val_loader)
        )

        val_visibility = (
            val_visibility_sum
            /
            len(val_loader)
        )

        # ====================================================
        # Scheduler
        # ====================================================

        scheduler.step(
            val_total
        )

        current_lr = optimizer.param_groups[0]["lr"]

        # ====================================================
        # RESULTS
        # ====================================================

        print("\n")

        print("=" * 70)

        print(
            f"EPOCH {epoch}/{EPOCHS} RESULTS"
        )

        print("=" * 70)

        print(
            f"Training Total Loss: "
            f"{train_total:.6f}"
        )

        print(
            f"Training Heatmap Loss: "
            f"{train_heatmap:.6f}"
        )

        print(
            f"Training Visibility BCE: "
            f"{train_visibility:.6f}"
        )

        print()

        print(
            f"Validation Total Loss: "
            f"{val_total:.6f}"
        )

        print(
            f"Validation Heatmap Loss: "
            f"{val_heatmap:.6f}"
        )

        print(
            f"Validation Visibility BCE: "
            f"{val_visibility:.6f}"
        )

        print()

        print(
            f"Learning rate: "
            f"{current_lr:.8f}"
        )

        # ====================================================
        # Save best model
        # ====================================================

        if val_total < best_val_loss:

            best_val_loss = val_total

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_total,
                "train_loss": train_total,
                "val_heatmap_loss": val_heatmap,
                "val_visibility_loss": val_visibility,
            }

            torch.save(
                checkpoint,
                BEST_MODEL_PATH
            )

            print()

            print(
                "*** NEW BEST MODEL SAVED ***"
            )

            print(
                "Path:",
                BEST_MODEL_PATH
            )

        else:

            print(
                "\nNo improvement in validation loss."
            )

    # ========================================================
    # Save final model
    # ========================================================

    torch.save(
        {
            "epoch": EPOCHS,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_total,
            "train_loss": train_total,
        },
        FINAL_MODEL_PATH
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")

    print("=" * 70)

    print(
        "EXPERIMENT 3 TRAINING COMPLETED"
    )

    print("=" * 70)

    print(
        f"\nBest validation loss: "
        f"{best_val_loss:.6f}"
    )

    print("\nBest model:")

    print(
        os.path.abspath(
            BEST_MODEL_PATH
        )
    )

    print("\nFinal model:")

    print(
        os.path.abspath(
            FINAL_MODEL_PATH
        )
    )

    print("\n" + "=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
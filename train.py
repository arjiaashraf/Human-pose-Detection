# ============================================================
# COCO HUMAN POSE TRAINING - EXPERIMENT 11
# ============================================================

import os
import json

import torch

from torch.utils.data import DataLoader

from src.config import *

from src.utils import (
    seed_everything,
    ensure_dirs,
    get_device,
    create_scaler,
    autocast_context,
)

from src.dataset import (
    COCOPoseDataset,
    load_coco_records,
    split_records,
)

from src.model import UNetPose

from src.loss import PoseLoss


# ============================================================
# DATALOADER
# ============================================================

def make_loader(
    dataset,
    batch_size,
    shuffle
):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(
            PERSISTENT_WORKERS
            and NUM_WORKERS > 0
        ),
        drop_last=False,
    )


# ============================================================
# MOVE BATCH
# ============================================================

def move_batch(
    batch,
    device
):

    (
        images,
        heatmaps,
        coords,
        visibility,
        visible,
        meta,
    ) = batch

    images = images.to(
        device,
        non_blocking=True
    )

    heatmaps = heatmaps.to(
        device,
        non_blocking=True
    )

    coords = coords.to(
        device,
        non_blocking=True
    )

    visibility = visibility.to(
        device,
        non_blocking=True
    )

    visible = visible.to(
        device,
        non_blocking=True
    )

    return (
        images,
        heatmaps,
        coords,
        visibility,
        visible,
        meta
    )


# ============================================================
# RUN EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    training=True
):

    if training:

        model.train()

    else:

        model.eval()

    totals = {
        "total": 0.0,
        "heatmap": 0.0,
        "coordinate": 0.0,
        "visibility": 0.0,
    }

    sample_count = 0

    for batch_idx, batch in enumerate(
        loader,
        start=1
    ):

        (
            images,
            target_heatmaps,
            target_coords,
            target_visibility,
            visible,
            _
        ) = move_batch(
            batch,
            device
        )

        if training:

            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            training
        ):

            with autocast_context(
                device
            ):

                (
                    pred_heatmaps,
                    pred_visibility
                ) = model(images)

                (
                    loss,
                    heatmap_loss,
                    coordinate_loss,
                    visibility_loss,
                    _
                ) = criterion(
                    pred_heatmaps,
                    pred_visibility,
                    target_heatmaps,
                    target_coords,
                    target_visibility,
                    visible,
                )

            # ------------------------------------------------
            # BACKPROP
            # ------------------------------------------------

            if training:

                if scaler is not None:

                    scaler.scale(
                        loss
                    ).backward()

                    scaler.unscale_(
                        optimizer
                    )

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=5.0
                    )

                    scaler.step(
                        optimizer
                    )

                    scaler.update()

                else:

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=5.0
                    )

                    optimizer.step()

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        current_batch_size = (
            images.size(0)
        )

        totals["total"] += (
            loss.detach().item()
            *
            current_batch_size
        )

        totals["heatmap"] += (
            heatmap_loss.detach().item()
            *
            current_batch_size
        )

        totals["coordinate"] += (
            coordinate_loss.detach().item()
            *
            current_batch_size
        )

        totals["visibility"] += (
            visibility_loss.detach().item()
            *
            current_batch_size
        )

        sample_count += current_batch_size

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if training:

            if (
                batch_idx == 1
                or batch_idx % 50 == 0
                or batch_idx == len(loader)
            ):

                print(
                    f"Batch "
                    f"[{batch_idx}/{len(loader)}] "
                    f"Total: {loss.item():.5f} | "
                    f"HM: {heatmap_loss.item():.5f} | "
                    f"COORD: "
                    f"{coordinate_loss.item():.5f} | "
                    f"VIS: "
                    f"{visibility_loss.item():.5f}"
                )

    denominator = max(
        sample_count,
        1
    )

    return {
        key: value / denominator
        for key, value in totals.items()
    }


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    metrics
):

    checkpoint = {

        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict()
            if optimizer is not None
            else None,

        "scheduler_state_dict":
            scheduler.state_dict()
            if scheduler is not None
            else None,

        "metrics":
            metrics,

        "config": {

            "image_size":
                IMAGE_SIZE,

            "heatmap_size":
                HEATMAP_SIZE,

            "num_keypoints":
                NUM_KEYPOINTS,

            "bbox_scale":
                BBOX_SCALE,

            "sigma":
                SIGMA,

            "softargmax_beta":
                SOFTARGMAX_BETA,
        }
    }

    torch.save(
        checkpoint,
        path
    )


# ============================================================
# TRAIN
# ============================================================

def train():

    seed_everything(
        SEED
    )

    ensure_dirs(
        OUTPUT_DIR,
        EVAL_DIR
    )

    device = get_device()

    print("=" * 72)
    print(
        "COCO HUMAN POSE TRAINING - "
        "EXPERIMENT 11"
    )
    print("=" * 72)

    print()

    print(
        "Device:",
        device
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "GPU memory:",
            round(
                torch.cuda.get_device_properties(
                    0
                ).total_memory
                /
                1024**3,
                2
            ),
            "GB"
        )

        print(
            "AMP: True"
        )

    else:

        print(
            "GPU: None"
        )

        print(
            "AMP: False"
        )

    # ========================================================
    # CONFIG
    # ========================================================

    print()
    print("=" * 72)
    print("CONFIGURATION")
    print("=" * 72)

    print(
        "Image size:",
        IMAGE_SIZE
    )

    print(
        "Heatmap size:",
        HEATMAP_SIZE
    )

    print(
        "Samples:",
        MAX_SAMPLES
    )

    print(
        "Batch:",
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
        "Weight decay:",
        WEIGHT_DECAY
    )

    print(
        "BBox scale:",
        BBOX_SCALE
    )

    print(
        "Sigma:",
        SIGMA
    )

    print(
        "Coordinate weight:",
        COORD_WEIGHT
    )

    # ========================================================
    # PATH CHECK
    # ========================================================

    if not os.path.isdir(
        IMAGE_DIR
    ):

        raise FileNotFoundError(
            f"Image directory not found:\n"
            f"{IMAGE_DIR}"
        )

    if not os.path.isfile(
        ANNOTATION_FILE
    ):

        raise FileNotFoundError(
            f"Annotation file not found:\n"
            f"{ANNOTATION_FILE}"
        )

    # ========================================================
    # DATA
    # ========================================================

    records = load_coco_records(
        ANNOTATION_FILE,
        IMAGE_DIR
    )

    train_records, val_records = (
        split_records(records)
    )

    train_dataset = COCOPoseDataset(
        train_records,
        IMAGE_DIR,
        IMAGE_SIZE,
        HEATMAP_SIZE,
        SIGMA,
        True,
        BBOX_SCALE
    )

    val_dataset = COCOPoseDataset(
        val_records,
        IMAGE_DIR,
        IMAGE_SIZE,
        HEATMAP_SIZE,
        SIGMA,
        False,
        BBOX_SCALE
    )

    train_loader = make_loader(
        train_dataset,
        BATCH_SIZE,
        True
    )

    val_loader = make_loader(
        val_dataset,
        BATCH_SIZE,
        False
    )

    print()
    print(
        "Training batches:",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(val_loader)
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = UNetPose(
        NUM_KEYPOINTS
    ).to(device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print(
        "Model parameters:",
        f"{parameters:,}"
    )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = PoseLoss(
        HEATMAP_WEIGHT,
        COORD_WEIGHT,
        VIS_WEIGHT,
        COORD_HUBER_BETA,
        SOFTARGMAX_BETA
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=MIN_LEARNING_RATE
    )

    scaler = create_scaler(
        device
    )

    # ========================================================
    # BEST
    # ========================================================

    best_coordinate_loss = float(
        "inf"
    )

    best_epoch = -1

    history = []

    # ========================================================
    # TRAINING
    # ========================================================

    print()
    print("=" * 72)
    print("STARTING TRAINING")
    print("=" * 72)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print()
        print(
            f"Epoch [{epoch}/{EPOCHS}]"
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            True
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            None,
            None,
            device,
            False
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        history.append(
            {
                "epoch":
                    epoch,

                "train":
                    train_metrics,

                "val":
                    val_metrics,

                "lr":
                    current_lr
            }
        )

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        print()
        print("=" * 72)
        print(
            f"EPOCH {epoch}/{EPOCHS}"
        )
        print("=" * 72)

        print(
            f"Train total: "
            f"{train_metrics['total']:.6f}"
        )

        print(
            f"Train heatmap: "
            f"{train_metrics['heatmap']:.6f}"
        )

        print(
            f"Train coordinate: "
            f"{train_metrics['coordinate']:.6f}"
        )

        print(
            f"Train visibility: "
            f"{train_metrics['visibility']:.6f}"
        )

        print()

        print(
            f"Val total: "
            f"{val_metrics['total']:.6f}"
        )

        print(
            f"Val heatmap: "
            f"{val_metrics['heatmap']:.6f}"
        )

        print(
            f"Val coordinate: "
            f"{val_metrics['coordinate']:.6f}"
        )

        print(
            f"Val visibility: "
            f"{val_metrics['visibility']:.6f}"
        )

        print()

        print(
            f"Learning rate: "
            f"{current_lr:.8f}"
        )

        # ----------------------------------------------------
        # BEST
        # ----------------------------------------------------

        if (
            val_metrics["coordinate"]
            <
            best_coordinate_loss
        ):

            best_coordinate_loss = (
                val_metrics["coordinate"]
            )

            best_epoch = epoch

            best_path = os.path.join(
                OUTPUT_DIR,
                "best_unet_pose.pth"
            )

            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch,
                val_metrics
            )

            print()
            print(
                "*** NEW BEST MODEL ***"
            )

            print(
                "Best coordinate loss:",
                best_coordinate_loss
            )

        # ----------------------------------------------------
        # LATEST
        # ----------------------------------------------------

        if SAVE_LAST_EVERY_EPOCH:

            latest_path = os.path.join(
                OUTPUT_DIR,
                "latest_unet_pose.pth"
            )

            save_checkpoint(
                latest_path,
                model,
                optimizer,
                scheduler,
                epoch,
                val_metrics
            )

        # ----------------------------------------------------
        # SCHEDULER
        # ----------------------------------------------------

        scheduler.step()

        # ----------------------------------------------------
        # GPU MEMORY
        # ----------------------------------------------------

        if device.type == "cuda":

            allocated = (
                torch.cuda.memory_allocated()
                /
                1024**3
            )

            reserved = (
                torch.cuda.memory_reserved()
                /
                1024**3
            )

            print(
                f"GPU memory: "
                f"{allocated:.2f} GB allocated | "
                f"{reserved:.2f} GB reserved"
            )

    # ========================================================
    # FINAL
    # ========================================================

    final_path = os.path.join(
        OUTPUT_DIR,
        "unet_pose_final.pth"
    )

    save_checkpoint(
        final_path,
        model,
        optimizer,
        scheduler,
        EPOCHS,
        history[-1]["val"]
    )

    history_path = os.path.join(
        OUTPUT_DIR,
        "training_history.json"
    )

    with open(
        history_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            indent=2
        )

    print()
    print("=" * 72)
    print(
        "EXPERIMENT 11 TRAINING COMPLETED"
    )
    print("=" * 72)

    print(
        "Best coordinate loss:",
        best_coordinate_loss
    )

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Best model:",
        os.path.join(
            OUTPUT_DIR,
            "best_unet_pose.pth"
        )
    )

    print(
        "Final model:",
        final_path
    )

    print(
        "History:",
        history_path
    )

    print("=" * 72)


if __name__ == "__main__":

    train()
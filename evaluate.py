# ============================================================
# COCO HUMAN POSE EVALUATION - EXPERIMENT 11
# ============================================================

import os
import math

import numpy as np

import torch

from src.config import *

from src.utils import (
    seed_everything,
    ensure_dirs,
    get_device,
    autocast_context,
)

from src.dataset import (
    COCOPoseDataset,
    load_coco_records,
    split_records,
)

from src.model import (
    UNetPose,
    SoftArgmax2D,
)

from train import (
    make_loader,
    move_batch,
)


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_model():

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
        "COCO POSE MODEL EVALUATION - "
        "EXPERIMENT 11"
    )
    print("=" * 72)

    print(
        "Device:",
        device
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ========================================================
    # DATA
    # ========================================================

    records = load_coco_records(
        ANNOTATION_FILE,
        IMAGE_DIR
    )

    _, val_records = split_records(
        records
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

    val_loader = make_loader(
        val_dataset,
        EVAL_BATCH_SIZE,
        False
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = UNetPose(
        NUM_KEYPOINTS
    ).to(device)

    checkpoint_path = os.path.join(
        OUTPUT_DIR,
        "best_unet_pose.pth"
    )

    if not os.path.isfile(
        checkpoint_path
    ):

        raise FileNotFoundError(
            f"Best checkpoint not found:\n"
            f"{checkpoint_path}\n"
            f"Run training first."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        checkpoint_epoch = checkpoint.get(
            "epoch",
            "?"
        )

        checkpoint_metrics = (
            checkpoint.get(
                "metrics",
                {}
            )
        )

    else:

        model.load_state_dict(
            checkpoint
        )

        checkpoint_epoch = "?"

        checkpoint_metrics = {}

    model.eval()

    softargmax = SoftArgmax2D(
        SOFTARGMAX_BETA
    ).to(device)

    print(
        "Checkpoint epoch:",
        checkpoint_epoch
    )

    print(
        "Checkpoint coordinate loss:",
        checkpoint_metrics.get(
            "coordinate",
            "N/A"
        )
    )

    # ========================================================
    # METRICS
    # ========================================================

    errors_hm = []

    errors_img = []

    per_joint_errors = [
        []
        for _ in range(NUM_KEYPOINTS)
    ]

    correct = {
        t: [0] * NUM_KEYPOINTS
        for t in (
            0.05,
            0.10,
            0.15,
            0.20
        )
    }

    counts = [
        0
        for _ in range(NUM_KEYPOINTS)
    ]

    vis_correct = 0

    vis_total = 0

    bbox_correct = {
        t: 0
        for t in (
            0.05,
            0.10,
            0.15,
            0.20
        )
    }

    bbox_total = 0

    # ========================================================
    # LOOP
    # ========================================================

    for batch_idx, batch in enumerate(
        val_loader,
        1
    ):

        (
            images,
            _,
            target_coords,
            target_visibility,
            visible,
            meta
        ) = move_batch(
            batch,
            device
        )

        with autocast_context(
            device
        ):

            (
                pred_heatmaps,
                pred_visibility_logits
            ) = model(images)

        # ----------------------------------------------------
        # SOFTARGMAX
        # ----------------------------------------------------

        pred_coords = softargmax(
            pred_heatmaps
        )

        # ----------------------------------------------------
        # VISIBILITY
        # ----------------------------------------------------

        vis_pred = (
            torch.sigmoid(
                pred_visibility_logits
            )
            >= 0.5
        ).float()

        vis_correct += int(
            (
                vis_pred
                ==
                target_visibility
            ).sum().item()
        )

        vis_total += int(
            target_visibility.numel()
        )

        # ----------------------------------------------------
        # EACH IMAGE
        # ----------------------------------------------------

        for b in range(
            images.shape[0]
        ):

            crop_w = float(
                meta["crop_w"][b]
            )

            crop_h = float(
                meta["crop_h"][b]
            )

            crop_x1 = float(
                meta["crop_x1"][b]
            )

            crop_y1 = float(
                meta["crop_y1"][b]
            )

            pred_xy = (
                pred_coords[b]
                .float()
                .cpu()
                .numpy()
            )

            target_xy = (
                target_coords[b]
                .float()
                .cpu()
                .numpy()
            )

            vis = (
                visible[b]
                .cpu()
                .numpy()
            )

            original_kps = (
                meta[
                    "original_keypoints"
                ][b]
                .numpy()
            )

            bbox_size = max(
                float(
                    meta["bbox_w"][b]
                ),
                float(
                    meta["bbox_h"][b]
                ),
                1.0
            )

            # ------------------------------------------------
            # KEYPOINTS
            # ------------------------------------------------

            for j in range(
                NUM_KEYPOINTS
            ):

                # Only evaluate visible COCO joints.
                if vis[j] < 0.5:

                    continue

                dx_hm = (
                    pred_xy[j, 0]
                    -
                    target_xy[j, 0]
                ) * (
                    HEATMAP_SIZE - 1
                )

                dy_hm = (
                    pred_xy[j, 1]
                    -
                    target_xy[j, 1]
                ) * (
                    HEATMAP_SIZE - 1
                )

                err_hm = math.hypot(
                    dx_hm,
                    dy_hm
                )

                # ------------------------------------------------
                # ORIGINAL IMAGE COORDINATES
                # ------------------------------------------------

                px = (
                    crop_x1
                    +
                    pred_xy[j, 0]
                    *
                    crop_w
                )

                py = (
                    crop_y1
                    +
                    pred_xy[j, 1]
                    *
                    crop_h
                )

                err_img = math.hypot(
                    px
                    -
                    float(
                        original_kps[j, 0]
                    ),

                    py
                    -
                    float(
                        original_kps[j, 1]
                    )
                )

                errors_hm.append(
                    err_hm
                )

                errors_img.append(
                    err_img
                )

                per_joint_errors[j].append(
                    err_hm
                )

                counts[j] += 1

                # ------------------------------------------------
                # PCK
                # ------------------------------------------------

                for t in correct:

                    correct[t][j] += int(
                        err_hm
                        <=
                        t
                        *
                        HEATMAP_SIZE
                    )

                # ------------------------------------------------
                # BBOX PCK
                # ------------------------------------------------

                norm_error = (
                    err_img
                    /
                    bbox_size
                )

                bbox_total += 1

                for t in bbox_correct:

                    bbox_correct[t] += int(
                        norm_error <= t
                    )

        if (
            batch_idx % 10 == 0
            or batch_idx == len(val_loader)
        ):

            print(
                f"Evaluated batch "
                f"{batch_idx}/"
                f"{len(val_loader)}"
            )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    eh = np.asarray(
        errors_hm,
        dtype=np.float32
    )

    ei = np.asarray(
        errors_img,
        dtype=np.float32
    )

    print()
    print("=" * 72)
    print(
        "FINAL EVALUATION RESULTS"
    )
    print("=" * 72)

    print(
        "Total evaluated keypoints:",
        len(eh)
    )

    print(
        f"Mean error: "
        f"{eh.mean():.4f} heatmap px "
        f"({ei.mean():.2f} image px)"
    )

    print(
        f"Median error: "
        f"{np.median(eh):.4f} heatmap px "
        f"({np.median(ei):.2f} image px)"
    )

    print(
        f"P90 error: "
        f"{np.percentile(eh, 90):.4f} "
        f"heatmap px"
    )

    print(
        f"P95 error: "
        f"{np.percentile(eh, 95):.4f} "
        f"heatmap px"
    )

    print(
        f"Visibility accuracy: "
        f"{100 * vis_correct / max(vis_total, 1):.2f}%"
    )

    # ========================================================
    # PCK
    # ========================================================

    print()
    print(
        "PCK RESULTS"
    )

    for t in correct:

        threshold = (
            t
            *
            HEATMAP_SIZE
        )

        score = (
            np.mean(
                eh <= threshold
            )
            *
            100.0
        )

        print(
            f"PCK @ {t:.2f}: "
            f"{score:.2f}%"
        )

    # ========================================================
    # BBOX PCK
    # ========================================================

    print()
    print(
        "BBOX-NORMALIZED PCK"
    )

    for t in bbox_correct:

        score = (
            100.0
            *
            bbox_correct[t]
            /
            max(
                bbox_total,
                1
            )
        )

        print(
            f"BBox PCK @ {t:.2f}: "
            f"{score:.2f}%"
        )

    # ========================================================
    # PER KEYPOINT
    # ========================================================

    print()
    print(
        "PER-KEYPOINT RESULTS"
    )

    print(
        f"{'ID':>2} "
        f"{'Keypoint':<17} "
        f"{'Error(HM)':>10} "
        f"{'PCK@.05':>9} "
        f"{'PCK@.10':>9} "
        f"{'PCK@.15':>9} "
        f"{'PCK@.20':>9} "
        f"{'Count':>8}"
    )

    for j, name in enumerate(
        COCO_NAMES
    ):

        if counts[j] == 0:

            continue

        arr = np.asarray(
            per_joint_errors[j],
            dtype=np.float32
        )

        c = counts[j]

        print(
            f"{j:2d} "
            f"{name:<17} "
            f"{arr.mean():10.4f} "
            f"{100 * correct[0.05][j] / c:9.2f} "
            f"{100 * correct[0.10][j] / c:9.2f} "
            f"{100 * correct[0.15][j] / c:9.2f} "
            f"{100 * correct[0.20][j] / c:9.2f} "
            f"{c:8d}"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    result_path = os.path.join(
        EVAL_DIR,
        "evaluation_results.txt"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "COCO POSE MODEL EVALUATION - "
            "EXPERIMENT 11\n"
        )

        f.write(
            f"Checkpoint epoch: "
            f"{checkpoint_epoch}\n"
        )

        f.write(
            f"Total evaluated keypoints: "
            f"{len(eh)}\n"
        )

        f.write(
            f"Mean error HM: "
            f"{eh.mean():.6f}\n"
        )

        f.write(
            f"Mean error IMG: "
            f"{ei.mean():.6f}\n"
        )

        f.write(
            f"Median error HM: "
            f"{np.median(eh):.6f}\n"
        )

        f.write(
            f"Median error IMG: "
            f"{np.median(ei):.6f}\n"
        )

        f.write(
            f"Visibility accuracy: "
            f"{100 * vis_correct / max(vis_total, 1):.4f}%\n"
        )

        for t in correct:

            score = (
                100
                *
                np.mean(
                    eh
                    <=
                    t * HEATMAP_SIZE
                )
            )

            f.write(
                f"PCK@{t:.2f}: "
                f"{score:.4f}%\n"
            )

        for t in bbox_correct:

            score = (
                100
                *
                bbox_correct[t]
                /
                max(
                    bbox_total,
                    1
                )
            )

            f.write(
                f"BBox PCK@{t:.2f}: "
                f"{score:.4f}%\n"
            )

    print()
    print(
        "Results saved to:",
        result_path
    )

    print("=" * 72)


if __name__ == "__main__":

    evaluate_model()
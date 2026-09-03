import random

from pathlib import Path

import numpy as np

import torch


# ============================================================
# RANDOM SEED
# ============================================================

def seed_everything(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    # Better training speed on GPU.
    torch.backends.cudnn.deterministic = False

    torch.backends.cudnn.benchmark = True


# ============================================================
# DIRECTORIES
# ============================================================

def ensure_dirs(output_dir, eval_dir):

    Path(output_dir).mkdir(
        parents=True,
        exist_ok=True
    )

    Path(eval_dir).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# DEVICE
# ============================================================

def get_device():

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# ============================================================
# SYSTEM INFORMATION
# ============================================================

def print_system_info(device):

    print("=" * 72)
    print("COCO HUMAN POSE")
    print("=" * 72)

    print()

    print("Device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print("CUDA available: YES")

        print(
            "CUDA version:",
            torch.version.cuda
        )

        props = torch.cuda.get_device_properties(0)

        print(
            "GPU memory:",
            round(
                props.total_memory / 1024**3,
                2
            ),
            "GB"
        )

        print("AMP: YES")

    else:

        print("CUDA available: NO")

        print("WARNING: Training on CPU will be slow.")

        print("AMP: NO")

    print()


# ============================================================
# BBOX CONVERSION
# ============================================================

def xywh_to_xyxy(box):

    x, y, w, h = [
        float(v)
        for v in box
    ]

    return (
        x,
        y,
        x + w,
        y + h
    )


# ============================================================
# SQUARE BBOX
# ============================================================

def make_square_bbox(
    box_xyxy,
    image_w,
    image_h,
    scale=1.20
):

    x1, y1, x2, y2 = box_xyxy

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    w = max(
        x2 - x1,
        2.0
    )

    h = max(
        y2 - y1,
        2.0
    )

    side = max(w, h) * scale

    nx1 = cx - side / 2.0
    ny1 = cy - side / 2.0

    nx2 = cx + side / 2.0
    ny2 = cy + side / 2.0

    # Keep crop inside image.
    if nx1 < 0:
        nx2 -= nx1
        nx1 = 0

    if ny1 < 0:
        ny2 -= ny1
        ny1 = 0

    if nx2 > image_w:
        shift = nx2 - image_w
        nx1 -= shift
        nx2 = image_w

    if ny2 > image_h:
        shift = ny2 - image_h
        ny1 -= shift
        ny2 = image_h

    nx1 = max(0.0, nx1)
    ny1 = max(0.0, ny1)

    nx2 = min(
        float(image_w),
        nx2
    )

    ny2 = min(
        float(image_h),
        ny2
    )

    if nx2 <= nx1:
        nx2 = min(
            float(image_w),
            nx1 + 2.0
        )

    if ny2 <= ny1:
        ny2 = min(
            float(image_h),
            ny1 + 2.0
        )

    return (
        nx1,
        ny1,
        nx2,
        ny2
    )


# ============================================================
# GAUSSIAN HEATMAP
# ============================================================

def gaussian_heatmap(
    cx,
    cy,
    size,
    sigma
):

    yy, xx = torch.meshgrid(
        torch.arange(
            size,
            dtype=torch.float32
        ),
        torch.arange(
            size,
            dtype=torch.float32
        ),
        indexing="ij"
    )

    heatmap = torch.exp(
        -(
            (xx - cx) ** 2
            +
            (yy - cy) ** 2
        )
        /
        (2.0 * sigma * sigma)
    )

    return heatmap


# ============================================================
# AMP SCALER
# ============================================================

def create_scaler(device):

    if device.type != "cuda":

        return None

    try:

        from torch.amp import GradScaler

        return GradScaler("cuda")

    except Exception:

        return torch.cuda.amp.GradScaler()


# ============================================================
# AMP CONTEXT
# ============================================================

def autocast_context(device):

    if device.type != "cuda":

        from contextlib import nullcontext

        return nullcontext()

    try:

        from torch.amp import autocast

        return autocast(
            "cuda",
            dtype=torch.float16
        )

    except Exception:

        return torch.cuda.amp.autocast(
            dtype=torch.float16
        )
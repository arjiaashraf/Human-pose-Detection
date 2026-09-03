import os
import json
import random

import numpy as np

from PIL import Image, ImageEnhance

import torch

from torch.utils.data import Dataset

from src.config import (
    NUM_KEYPOINTS,
    MIN_VISIBLE_KEYPOINTS,
    MAX_SAMPLES,
    TRAIN_RATIO,
    SEED,
    HORIZONTAL_FLIP_PROB,
    COLOR_JITTER_PROB,
    BRIGHTNESS_RANGE,
    CONTRAST_RANGE,
    SATURATION_RANGE,
    HUE_RANGE,
    FLIP_PAIRS,
)

from src.utils import (
    xywh_to_xyxy,
    make_square_bbox,
    gaussian_heatmap,
)


class COCOPoseDataset(Dataset):

    def __init__(
        self,
        records,
        image_dir,
        image_size=384,
        heatmap_size=96,
        sigma=2.5,
        train=False,
        bbox_scale=1.20,
    ):

        self.records = records

        self.image_dir = image_dir

        self.image_size = image_size

        self.heatmap_size = heatmap_size

        self.sigma = sigma

        self.train = train

        self.bbox_scale = bbox_scale

    def __len__(self):

        return len(self.records)

    # ========================================================
    # COLOR AUGMENTATION
    # ========================================================

    def apply_color_augmentation(self, image):

        if random.random() > COLOR_JITTER_PROB:

            return image

        brightness = random.uniform(
            1.0 - BRIGHTNESS_RANGE,
            1.0 + BRIGHTNESS_RANGE
        )

        contrast = random.uniform(
            1.0 - CONTRAST_RANGE,
            1.0 + CONTRAST_RANGE
        )

        saturation = random.uniform(
            1.0 - SATURATION_RANGE,
            1.0 + SATURATION_RANGE
        )

        image = ImageEnhance.Brightness(
            image
        ).enhance(brightness)

        image = ImageEnhance.Contrast(
            image
        ).enhance(contrast)

        image = ImageEnhance.Color(
            image
        ).enhance(saturation)

        return image

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, idx):

        rec = self.records[idx]

        image_path = os.path.join(
            self.image_dir,
            rec["file_name"]
        )

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as e:

            raise RuntimeError(
                f"Could not open image: "
                f"{image_path}\nError: {e}"
            )

        image_w, image_h = image.size

        kps = np.array(
            rec["keypoints"],
            dtype=np.float32
        ).reshape(
            NUM_KEYPOINTS,
            3
        )

        original_kps = kps.copy()

        # ----------------------------------------------------
        # ORIGINAL BBOX
        # ----------------------------------------------------

        bx1, by1, bx2, by2 = xywh_to_xyxy(
            rec["bbox"]
        )

        bx1 = max(
            0.0,
            min(
                bx1,
                image_w - 1
            )
        )

        by1 = max(
            0.0,
            min(
                by1,
                image_h - 1
            )
        )

        bx2 = max(
            bx1 + 1.0,
            min(
                bx2,
                float(image_w)
            )
        )

        by2 = max(
            by1 + 1.0,
            min(
                by2,
                float(image_h)
            )
        )

        # ----------------------------------------------------
        # PERSON CROP
        # ----------------------------------------------------

        cx1, cy1, cx2, cy2 = make_square_bbox(
            (
                bx1,
                by1,
                bx2,
                by2
            ),
            image_w,
            image_h,
            self.bbox_scale
        )

        actual_x1 = int(
            round(cx1)
        )

        actual_y1 = int(
            round(cy1)
        )

        actual_x2 = int(
            round(cx2)
        )

        actual_y2 = int(
            round(cy2)
        )

        crop = image.crop(
            (
                actual_x1,
                actual_y1,
                actual_x2,
                actual_y2
            )
        )

        crop_w = max(
            actual_x2 - actual_x1,
            1
        )

        crop_h = max(
            actual_y2 - actual_y1,
            1
        )

        # ----------------------------------------------------
        # KEYPOINTS INTO CROP COORDINATES
        # ----------------------------------------------------

        kp = kps.copy()

        kp[:, 0] -= actual_x1
        kp[:, 1] -= actual_y1

        flipped = False

        # ----------------------------------------------------
        # HORIZONTAL FLIP
        # ----------------------------------------------------

        if (
            self.train
            and random.random()
            < HORIZONTAL_FLIP_PROB
        ):

            crop = crop.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )

            kp[:, 0] = (
                crop_w - 1
                - kp[:, 0]
            )

            for a, b in FLIP_PAIRS:

                kp[[a, b]] = kp[[b, a]]

            flipped = True

        # ----------------------------------------------------
        # COLOR AUGMENTATION
        # ----------------------------------------------------

        if self.train:

            crop = self.apply_color_augmentation(
                crop
            )

        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        crop = crop.resize(
            (
                self.image_size,
                self.image_size
            ),
            Image.Resampling.BILINEAR
        )

        # ----------------------------------------------------
        # IMAGE TENSOR
        # ----------------------------------------------------

        image_array = np.asarray(
            crop,
            dtype=np.float32
        ) / 255.0

        image_tensor = torch.from_numpy(
            image_array
        ).permute(
            2,
            0,
            1
        )

        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(
            3,
            1,
            1
        )

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(
            3,
            1,
            1
        )

        image_tensor = (
            image_tensor - mean
        ) / std

        # ----------------------------------------------------
        # VISIBILITY
        # ----------------------------------------------------

        visibility = torch.from_numpy(
            (
                kp[:, 2] > 0
            ).astype(
                np.float32
            )
        )

        visible = torch.from_numpy(
            (
                kp[:, 2] == 2
            ).astype(
                np.float32
            )
        )

        # ----------------------------------------------------
        # NORMALIZED COORDINATES
        # ----------------------------------------------------

        norm_x = kp[:, 0] / float(crop_w)

        norm_y = kp[:, 1] / float(crop_h)

        # ----------------------------------------------------
        # CLAMP COORDINATES
        # ----------------------------------------------------

        norm_x = np.clip(
            norm_x,
            0.0,
            1.0
        )

        norm_y = np.clip(
            norm_y,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # HEATMAP COORDINATES
        # ----------------------------------------------------

        heat_x = (
            norm_x
            *
            (self.heatmap_size - 1)
        )

        heat_y = (
            norm_y
            *
            (self.heatmap_size - 1)
        )

        # ----------------------------------------------------
        # HEATMAPS
        # ----------------------------------------------------

        heatmaps = torch.zeros(
            self.heatmap_size,
            self.heatmap_size,
            NUM_KEYPOINTS,
            dtype=torch.float32
        )

        for j in range(NUM_KEYPOINTS):

            if visibility[j].item() > 0:

                heatmaps[
                    :,
                    :,
                    j
                ] = gaussian_heatmap(
                    float(heat_x[j]),
                    float(heat_y[j]),
                    self.heatmap_size,
                    self.sigma
                )

        heatmaps = heatmaps.permute(
            2,
            0,
            1
        ).contiguous()

        # ----------------------------------------------------
        # COORDINATES
        # ----------------------------------------------------

        coords = torch.from_numpy(
            np.stack(
                [
                    norm_x,
                    norm_y
                ],
                axis=1
            ).astype(
                np.float32
            )
        )

        # ----------------------------------------------------
        # METADATA
        # ----------------------------------------------------

        meta = {

            "image_id":
                int(rec["image_id"]),

            "file_name":
                rec["file_name"],

            "orig_w":
                int(image_w),

            "orig_h":
                int(image_h),

            "crop_x1":
                int(actual_x1),

            "crop_y1":
                int(actual_y1),

            "crop_w":
                int(crop_w),

            "crop_h":
                int(crop_h),

            "bbox_w":
                float(
                    max(
                        bx2 - bx1,
                        1.0
                    )
                ),

            "bbox_h":
                float(
                    max(
                        by2 - by1,
                        1.0
                    )
                ),

            "original_keypoints":
                original_kps.astype(
                    np.float32
                ),

            "flipped":
                flipped,
        }

        return (
            image_tensor,
            heatmaps,
            coords,
            visibility,
            visible,
            meta
        )


# ============================================================
# LOAD COCO
# ============================================================

def load_coco_records(
    annotation_file,
    image_dir
):

    print("=" * 72)
    print("CREATING DATASET")
    print("=" * 72)

    if not os.path.isdir(image_dir):

        raise FileNotFoundError(
            f"Image directory does not exist:\n"
            f"{image_dir}"
        )

    if not os.path.isfile(annotation_file):

        raise FileNotFoundError(
            f"Annotation file does not exist:\n"
            f"{annotation_file}"
        )

    with open(
        annotation_file,
        "r",
        encoding="utf-8"
    ) as f:

        coco = json.load(f)

    images = {
        int(img["id"]): img
        for img in coco.get(
            "images",
            []
        )
    }

    records = []

    for ann in coco.get(
        "annotations",
        []
    ):

        if ann.get(
            "category_id"
        ) != 1:

            continue

        kps = ann.get(
            "keypoints"
        )

        if (
            not kps
            or len(kps) != 51
        ):

            continue

        visible_count = sum(
            1
            for i in range(17)
            if kps[i * 3 + 2] == 2
        )

        labeled_count = sum(
            1
            for i in range(17)
            if kps[i * 3 + 2] > 0
        )

        if (
            visible_count
            < MIN_VISIBLE_KEYPOINTS
        ):

            continue

        if (
            labeled_count
            < MIN_VISIBLE_KEYPOINTS
        ):

            continue

        bbox = ann.get(
            "bbox"
        )

        if (
            not bbox
            or bbox[2] <= 2
            or bbox[3] <= 2
        ):

            continue

        image_info = images.get(
            int(
                ann["image_id"]
            )
        )

        if image_info is None:

            continue

        file_name = image_info[
            "file_name"
        ]

        image_path = os.path.join(
            image_dir,
            file_name
        )

        if not os.path.isfile(
            image_path
        ):

            continue

        records.append(
            {
                "image_id":
                    int(
                        ann["image_id"]
                    ),

                "file_name":
                    file_name,

                "bbox":
                    [
                        float(x)
                        for x in bbox
                    ],

                "keypoints":
                    [
                        float(x)
                        for x in kps
                    ],
            }
        )

    print(
        "Usable person annotations:",
        len(records)
    )

    if MAX_SAMPLES is not None:

        records = records[
            :MAX_SAMPLES
        ]

        print(
            "Using maximum samples:",
            MAX_SAMPLES
        )

    print(
        "Full dataset size:",
        len(records)
    )

    print()

    if len(records) < 10:

        raise RuntimeError(
            "Too few usable annotations."
        )

    return records


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

def split_records(records):

    rng = random.Random(
        SEED
    )

    indices = list(
        range(
            len(records)
        )
    )

    rng.shuffle(
        indices
    )

    train_count = int(
        len(indices)
        *
        TRAIN_RATIO
    )

    train_records = [
        records[i]
        for i in indices[
            :train_count
        ]
    ]

    val_records = [
        records[i]
        for i in indices[
            train_count:
        ]
    ]

    print(
        "Training samples:",
        len(train_records)
    )

    print(
        "Validation samples:",
        len(val_records)
    )

    return (
        train_records,
        val_records
    )
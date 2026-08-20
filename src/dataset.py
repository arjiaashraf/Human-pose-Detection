import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class COCOKeypointDataset(Dataset):

    def __init__(
        self,
        image_dir,
        annotation_file,
        image_size=256,
        heatmap_size=64,
        sigma=2.0,
        padding=0.20,
    ):
        self.image_dir = image_dir
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.padding = padding

        # ----------------------------------------------------
        # Load COCO annotations
        # ----------------------------------------------------

        with open(annotation_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = {
            img["id"]: img
            for img in coco["images"]
        }

        # ----------------------------------------------------
        # Keep annotations containing keypoints
        # ----------------------------------------------------

        self.annotations = []

        for ann in coco["annotations"]:

            if ann.get("num_keypoints", 0) <= 0:
                continue

            if ann.get("category_id") != 1:
                continue

            image_id = ann["image_id"]

            if image_id not in self.images:
                continue

            self.annotations.append(ann)

        # ----------------------------------------------------
        # Keep only annotations whose images exist
        # ----------------------------------------------------

        usable_annotations = []

        for ann in self.annotations:

            image_info = self.images[ann["image_id"]]

            filename = image_info["file_name"]

            image_path = os.path.join(
                self.image_dir,
                filename
            )

            if os.path.exists(image_path):
                usable_annotations.append(ann)

        self.annotations = usable_annotations

        print(
            "Usable person annotations:",
            len(self.annotations)
        )

    # ========================================================
    # Dataset length
    # ========================================================

    def __len__(self):
        return len(self.annotations)

    # ========================================================
    # Get sample
    # ========================================================

    def __getitem__(self, idx):

        ann = self.annotations[idx]

        image_id = ann["image_id"]

        image_info = self.images[image_id]

        filename = image_info["file_name"]

        image_path = os.path.join(
            self.image_dir,
            filename
        )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(
                f"Could not load image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        original_h, original_w = image.shape[:2]

        # ----------------------------------------------------
        # Read bounding box
        #
        # COCO format:
        # [x, y, width, height]
        # ----------------------------------------------------

        x, y, w, h = ann["bbox"]

        x1 = float(x)
        y1 = float(y)
        x2 = float(x + w)
        y2 = float(y + h)

        # ----------------------------------------------------
        # Add padding around person
        # ----------------------------------------------------

        pad_x = w * self.padding
        pad_y = h * self.padding

        x1 -= pad_x
        y1 -= pad_y
        x2 += pad_x
        y2 += pad_y

        # Clamp to image boundaries

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(original_w, x2)
        y2 = min(original_h, y2)

        # Make integer crop coordinates

        x1_int = int(round(x1))
        y1_int = int(round(y1))
        x2_int = int(round(x2))
        y2_int = int(round(y2))

        # ----------------------------------------------------
        # Crop person
        # ----------------------------------------------------

        crop = image[
            y1_int:y2_int,
            x1_int:x2_int
        ]

        if crop.size == 0:
            raise RuntimeError(
                f"Empty crop for annotation {idx}"
            )

        crop_h, crop_w = crop.shape[:2]

        # ----------------------------------------------------
        # COCO keypoints
        # ----------------------------------------------------

        keypoints = np.array(
            ann["keypoints"],
            dtype=np.float32
        ).reshape(17, 3)

        # ----------------------------------------------------
        # Convert image coordinates
        # to crop coordinates
        # ----------------------------------------------------

        keypoints[:, 0] -= x1_int
        keypoints[:, 1] -= y1_int

        # ----------------------------------------------------
        # Resize crop
        # ----------------------------------------------------

        crop = cv2.resize(
            crop,
            (
                self.image_size,
                self.image_size
            ),
            interpolation=cv2.INTER_LINEAR
        )

        # ----------------------------------------------------
        # Scale keypoints
        # ----------------------------------------------------

        scale_x = self.image_size / crop_w
        scale_y = self.image_size / crop_h

        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

        # ----------------------------------------------------
        # Normalize image
        # ----------------------------------------------------

        crop = crop.astype(
            np.float32
        ) / 255.0

        # HWC → CHW

        crop = np.transpose(
            crop,
            (2, 0, 1)
        )

        image_tensor = torch.tensor(
            crop,
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Generate heatmaps
        # ----------------------------------------------------

        heatmaps = self.generate_heatmaps(
            keypoints
        )

        # ----------------------------------------------------
        # Visibility
        #
        # COCO:
        # 0 = not labeled
        # 1 = labeled but not visible
        # 2 = visible
        #
        # For training:
        # > 0 means keypoint has annotation
        # ----------------------------------------------------

        visibility = (
            keypoints[:, 2] > 0
        ).astype(np.float32)

        visibility = torch.tensor(
            visibility,
            dtype=torch.float32
        )

        return (
            image_tensor,
            heatmaps,
            visibility
        )

    # ========================================================
    # Generate Gaussian heatmaps
    # ========================================================

    def generate_heatmaps(self, keypoints):

        heatmaps = np.zeros(
            (
                17,
                self.heatmap_size,
                self.heatmap_size
            ),
            dtype=np.float32
        )

        scale = (
            self.heatmap_size
            / self.image_size
        )

        # Coordinate grid

        xx, yy = np.meshgrid(
            np.arange(self.heatmap_size),
            np.arange(self.heatmap_size)
        )

        for joint_id in range(17):

            x, y, visible = keypoints[joint_id]

            if visible == 0:
                continue

            # Convert to heatmap coordinates

            hx = x * scale
            hy = y * scale

            # Gaussian heatmap

            heatmap = np.exp(
                -(
                    (xx - hx) ** 2
                    + (yy - hy) ** 2
                )
                / (
                    2 * self.sigma ** 2
                )
            )

            heatmaps[joint_id] = heatmap

        return torch.tensor(
            heatmaps,
            dtype=torch.float32
        )
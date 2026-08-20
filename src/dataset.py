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
        sigma=2.5,
        padding=0.20,
    ):
        self.image_dir = image_dir
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.padding = padding

        with open(annotation_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = {
            img["id"]: img
            for img in coco["images"]
        }

        self.annotations = []

        for ann in coco["annotations"]:

            if ann.get("category_id") != 1:
                continue

            if ann.get("num_keypoints", 0) <= 0:
                continue

            image_id = ann["image_id"]

            if image_id not in self.images:
                continue

            image_info = self.images[image_id]

            image_path = os.path.join(
                self.image_dir,
                image_info["file_name"]
            )

            if os.path.exists(image_path):
                self.annotations.append(ann)

        print(
            "Usable person annotations:",
            len(self.annotations)
        )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        ann = self.annotations[idx]

        image_info = self.images[
            ann["image_id"]
        ]

        image_path = os.path.join(
            self.image_dir,
            image_info["file_name"]
        )

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(
                f"Could not load image:\n{image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        original_h, original_w = image.shape[:2]

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x, y, w, h = ann["bbox"]

        x1 = float(x)
        y1 = float(y)
        x2 = float(x + w)
        y2 = float(y + h)

        # Padding
        pad_x = w * self.padding
        pad_y = h * self.padding

        x1 -= pad_x
        y1 -= pad_y
        x2 += pad_x
        y2 += pad_y

        # Clamp
        x1 = max(0.0, x1)
        y1 = max(0.0, y1)

        x2 = min(float(original_w), x2)
        y2 = min(float(original_h), y2)

        # Ensure valid crop
        x1_int = int(np.floor(x1))
        y1_int = int(np.floor(y1))
        x2_int = int(np.ceil(x2))
        y2_int = int(np.ceil(y2))

        x2_int = max(x2_int, x1_int + 1)
        y2_int = max(y2_int, y1_int + 1)

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
        # Keypoints
        # ----------------------------------------------------

        keypoints = np.array(
            ann["keypoints"],
            dtype=np.float32
        ).reshape(17, 3)

        # Convert original image coordinates
        # into crop coordinates

        keypoints[:, 0] -= x1_int
        keypoints[:, 1] -= y1_int

        # ----------------------------------------------------
        # Resize image to 256x256
        # ----------------------------------------------------

        crop = cv2.resize(
            crop,
            (
                self.image_size,
                self.image_size
            ),
            interpolation=cv2.INTER_LINEAR
        )

        scale_x = (
            self.image_size / float(crop_w)
        )

        scale_y = (
            self.image_size / float(crop_h)
        )

        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

        # ----------------------------------------------------
        # Image tensor
        # ----------------------------------------------------

        crop = crop.astype(
            np.float32
        ) / 255.0

        crop = np.transpose(
            crop,
            (2, 0, 1)
        )

        image_tensor = torch.from_numpy(
            crop
        ).float()

        # ----------------------------------------------------
        # Heatmaps
        # ----------------------------------------------------

        heatmaps = self.generate_heatmaps(
            keypoints
        )

        # ----------------------------------------------------
        # Visibility
        # ----------------------------------------------------

        visibility = (
            keypoints[:, 2] > 0
        ).astype(np.float32)

        visibility = torch.from_numpy(
            visibility
        ).float()

        return (
            image_tensor,
            heatmaps,
            visibility
        )

    # ========================================================
    # Gaussian heatmaps
    # ========================================================

    def generate_heatmaps(self, keypoints):

        H = self.heatmap_size
        W = self.heatmap_size

        heatmaps = np.zeros(
            (17, H, W),
            dtype=np.float32
        )

        scale = (
            H / float(self.image_size)
        )

        xx, yy = np.meshgrid(
            np.arange(W, dtype=np.float32),
            np.arange(H, dtype=np.float32)
        )

        for joint_id in range(17):

            x, y, visibility = (
                keypoints[joint_id]
            )

            if visibility <= 0:
                continue

            hx = x * scale
            hy = y * scale

            # Ignore points far outside target
            if (
                hx < -3 * self.sigma
                or hx > W + 3 * self.sigma
                or hy < -3 * self.sigma
                or hy > H + 3 * self.sigma
            ):
                continue

            exponent = (
                (
                    (xx - hx) ** 2
                    +
                    (yy - hy) ** 2
                )
                /
                (2.0 * self.sigma ** 2)
            )

            heatmaps[joint_id] = np.exp(
                -exponent
            )

        return torch.from_numpy(
            heatmaps
        ).float()
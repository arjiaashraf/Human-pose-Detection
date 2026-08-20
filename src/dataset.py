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
    ):

        self.image_dir = image_dir
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma

        # --------------------------------------------------
        # Load COCO annotations
        # --------------------------------------------------

        with open(annotation_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # Map image ID -> image information
        self.images = {
            img["id"]: img
            for img in coco["images"]
        }

        # --------------------------------------------------
        # Find images that actually exist on disk
        # --------------------------------------------------

        available_images = set(
            os.listdir(self.image_dir)
        )

        print("Images found on disk:", len(available_images))

        # --------------------------------------------------
        # Keep only usable annotations
        # --------------------------------------------------

        self.annotations = []

        for ann in coco["annotations"]:

            # Must contain at least one keypoint
            if ann["num_keypoints"] <= 0:
                continue

            image_id = ann["image_id"]

            # Get image information
            if image_id not in self.images:
                continue

            filename = self.images[image_id]["file_name"]

            # Check whether image exists
            if filename not in available_images:
                continue

            self.annotations.append(ann)

        print(
            "Usable annotations with downloaded images:",
            len(self.annotations)
        )

    # --------------------------------------------------
    # Dataset length
    # --------------------------------------------------

    def __len__(self):
        return len(self.annotations)

    # --------------------------------------------------
    # Get one sample
    # --------------------------------------------------

    def __getitem__(self, idx):

        ann = self.annotations[idx]

        image_id = ann["image_id"]

        image_info = self.images[image_id]

        filename = image_info["file_name"]

        image_path = os.path.join(
            self.image_dir,
            filename
        )

        # --------------------------------------------------
        # Load image
        # --------------------------------------------------

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(
                f"Could not load image: {image_path}"
            )

        # BGR -> RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        original_h, original_w = image.shape[:2]

        # --------------------------------------------------
        # Resize image
        # --------------------------------------------------

        image = cv2.resize(
            image,
            (self.image_size, self.image_size)
        )

        # Normalize
        image = image.astype(
            np.float32
        ) / 255.0

        # HWC -> CHW
        image = np.transpose(
            image,
            (2, 0, 1)
        )

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        # --------------------------------------------------
        # COCO keypoints
        # --------------------------------------------------

        keypoints = np.array(
            ann["keypoints"],
            dtype=np.float32
        ).reshape(17, 3)

        # --------------------------------------------------
        # Scale coordinates
        # --------------------------------------------------

        scale_x = self.image_size / original_w
        scale_y = self.image_size / original_h

        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

        # --------------------------------------------------
        # Generate Gaussian heatmaps
        # --------------------------------------------------

        heatmaps = self.generate_heatmaps(
            keypoints
        )

        # --------------------------------------------------
        # Visibility
        # --------------------------------------------------

        visibility = (
            keypoints[:, 2] > 0
        ).astype(np.float32)

        visibility = torch.tensor(
            visibility,
            dtype=torch.float32
        )

        return image, heatmaps, visibility

    # --------------------------------------------------
    # Gaussian heatmap generation
    # --------------------------------------------------

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
            self.heatmap_size /
            self.image_size
        )

        # Create grid once
        xx, yy = np.meshgrid(
            np.arange(self.heatmap_size),
            np.arange(self.heatmap_size)
        )

        for joint_id in range(17):

            x, y, visible = keypoints[joint_id]

            if visible == 0:
                continue

            # Convert image coordinates
            # to heatmap coordinates
            x_h = x * scale
            y_h = y * scale

            # Gaussian
            heatmap = np.exp(
                -(
                    (xx - x_h) ** 2 +
                    (yy - y_h) ** 2
                )
                /
                (2 * self.sigma ** 2)
            )

            heatmaps[joint_id] = heatmap

        return torch.tensor(
            heatmaps,
            dtype=torch.float32
        )
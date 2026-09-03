# ============================================================
# COCO POSE INFERENCE - EXPERIMENT 11
# EXACT TRAINING PREPROCESSING
# ARBITRARY IMAGE + YOLO PERSON DETECTION
# ============================================================

import os
import json
import cv2
import numpy as np

import torch
import torch.nn.functional as F

from ultralytics import YOLO

from src.model import UNetPose


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "/content/drive/MyDrive/ai dataset"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "checkpoints",
    "experiment11",
    "best_unet_pose.pth"
)

IMAGE_NAME = "pose_result.jpg"

IMAGE_PATH = os.path.join(
    BASE_DIR,
    IMAGE_NAME
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "poseresult_experiment11.jpg"
)

# ============================================================
# COCO DATA
# ============================================================

ANNOTATION_FILE = (
    "/content/drive/MyDrive/fiftyone/coco-2017/raw/"
    "person_keypoints_train2017.json"
)

IMAGE_DIR = (
    "/content/drive/MyDrive/fiftyone/coco-2017/train/data"
)

# ============================================================
# EXPERIMENT 11 CONFIG
# ============================================================

IMAGE_SIZE = 384
HEATMAP_SIZE = 96
NUM_KEYPOINTS = 17

SIGMA = 2.5

BBOX_SCALE = 1.20

SOFTARGMAX_BETA = 50.0

VISIBILITY_THRESHOLD = 0.30

# ============================================================
# MANUAL BBOX
# ============================================================
#
# Leave as None to use:
#
# COCO annotation -> YOLO detection
#
# Format:
#
# [x, y, width, height]
#
# Example:
#
# MANUAL_BBOX = [260, 109, 171, 312]
#
# ============================================================

MANUAL_BBOX = None

# ============================================================
# YOLO
# ============================================================

YOLO_MODEL = "yolo11n.pt"

YOLO_CONFIDENCE = 0.25

PERSON_CLASS_ID = 0

# ------------------------------------------------------------
# If multiple people are detected:
#
# "largest" = largest detected person
# "confidence" = highest confidence
# ------------------------------------------------------------

PERSON_SELECTION = "largest"

# ============================================================
# IMAGE NET NORMALIZATION
# EXACTLY SAME AS DATASET
# ============================================================

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)

# ============================================================
# COCO KEYPOINT NAMES
# ============================================================

KEYPOINT_NAMES = [

    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",

    "left_shoulder",
    "right_shoulder",

    "left_elbow",
    "right_elbow",

    "left_wrist",
    "right_wrist",

    "left_hip",
    "right_hip",

    "left_knee",
    "right_knee",

    "left_ankle",
    "right_ankle"
]

# ============================================================
# COCO SKELETON
# ============================================================

SKELETON = [

    (0, 1),
    (0, 2),

    (1, 3),
    (2, 4),

    (5, 6),

    (5, 7),
    (7, 9),

    (6, 8),
    (8, 10),

    (5, 11),
    (6, 12),

    (11, 12),

    (11, 13),
    (13, 15),

    (12, 14),
    (14, 16),

    (0, 5),
    (0, 6)
]

# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MAKE SQUARE BBOX
#
# THIS IS COPIED FROM YOUR TRAINING UTIL:
#
# src.utils.make_square_bbox
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

    side = max(
        w,
        h
    ) * scale

    nx1 = max(
        0.0,
        cx - side / 2.0
    )

    ny1 = max(
        0.0,
        cy - side / 2.0
    )

    nx2 = min(
        float(image_w),
        cx + side / 2.0
    )

    ny2 = min(
        float(image_h),
        cy + side / 2.0
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
# FIND COCO BBOX
# ============================================================

def find_coco_bbox(
    image_path,
    annotation_file
):

    print()
    print("=" * 72)
    print("SEARCHING COCO ANNOTATIONS")
    print("=" * 72)

    if not os.path.isfile(annotation_file):

        print(
            "COCO annotation file not found."
        )

        return None

    filename = os.path.basename(
        image_path
    )

    print()
    print(
        "Looking for:",
        filename
    )

    try:

        with open(
            annotation_file,
            "r",
            encoding="utf-8"
        ) as f:

            coco = json.load(f)

    except Exception as e:

        print(
            "Could not read annotation file:"
        )

        print(e)

        return None

    images = coco.get(
        "images",
        []
    )

    image_id = None

    for image_info in images:

        if image_info.get(
            "file_name"
        ) == filename:

            image_id = int(
                image_info["id"]
            )

            break

    if image_id is None:

        print(
            "Image filename was not found "
            "in COCO annotations."
        )

        return None

    print(
        "COCO image ID:",
        image_id
    )

    candidates = []

    for ann in coco.get(
        "annotations",
        []
    ):

        if int(
            ann.get(
                "image_id",
                -1
            )
        ) != image_id:

            continue

        if int(
            ann.get(
                "category_id",
                -1
            )
        ) != 1:

            continue

        bbox = ann.get(
            "bbox"
        )

        if (
            bbox is None
            or len(bbox) != 4
        ):

            continue

        x, y, w, h = [
            float(v)
            for v in bbox
        ]

        if w <= 2 or h <= 2:
            continue

        area = w * h

        candidates.append(
            (
                area,
                [x, y, w, h]
            )
        )

    if not candidates:

        print(
            "No valid person bbox found."
        )

        return None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    bbox = candidates[0][1]

    print()
    print(
        "COCO person bbox:"
    )

    print(
        [round(v, 2) for v in bbox]
    )

    return bbox


# ============================================================
# YOLO PERSON DETECTION
# ============================================================

def detect_person_bbox(
    image_path
):

    print()
    print("=" * 72)
    print("YOLO PERSON DETECTION")
    print("=" * 72)

    print()
    print(
        "Loading YOLO:",
        YOLO_MODEL
    )

    detector = YOLO(
        YOLO_MODEL
    )

    results = detector.predict(
        source=image_path,
        conf=YOLO_CONFIDENCE,
        classes=[
            PERSON_CLASS_ID
        ],
        verbose=False
    )

    if not results:

        raise RuntimeError(
            "YOLO returned no results."
        )

    result = results[0]

    if (
        result.boxes is None
        or len(result.boxes) == 0
    ):

        raise RuntimeError(
            "\nYOLO did not detect "
            "any person."
        )

    boxes = (
        result
        .boxes
        .xyxy
        .detach()
        .cpu()
        .numpy()
    )

    confidences = (
        result
        .boxes
        .conf
        .detach()
        .cpu()
        .numpy()
    )

    print()
    print(
        "Persons detected:",
        len(boxes)
    )

    areas = []

    for i, box in enumerate(
        boxes
    ):

        x1, y1, x2, y2 = box

        width = max(
            x2 - x1,
            1.0
        )

        height = max(
            y2 - y1,
            1.0
        )

        area = (
            width
            *
            height
        )

        areas.append(area)

        print(
            f"Person {i}: "
            f"bbox=("
            f"{x1:.1f}, "
            f"{y1:.1f}, "
            f"{x2:.1f}, "
            f"{y2:.1f}) "
            f"confidence="
            f"{confidences[i]:.3f} "
            f"area="
            f"{area:.1f}"
        )

    # --------------------------------------------------------
    # Select person
    # --------------------------------------------------------

    if PERSON_SELECTION == "confidence":

        selected_index = int(
            np.argmax(confidences)
        )

    else:

        selected_index = int(
            np.argmax(
                np.array(areas)
            )
        )

    x1, y1, x2, y2 = (
        boxes[selected_index]
    )

    bbox = [

        float(x1),

        float(y1),

        float(x2 - x1),

        float(y2 - y1)
    ]

    print()
    print(
        "Selected person:",
        selected_index
    )

    print(
        "BBox [x, y, width, height]:"
    )

    print(
        [round(v, 2) for v in bbox]
    )

    print(
        "Confidence:",
        f"{confidences[selected_index]:.3f}"
    )

    return bbox


# ============================================================
# GET BBOX
# ============================================================

def get_bbox(
    image_path,
    image_w,
    image_h
):

    # --------------------------------------------------------
    # 1. Manual
    # --------------------------------------------------------

    if MANUAL_BBOX is not None:

        if len(MANUAL_BBOX) != 4:

            raise ValueError(
                "MANUAL_BBOX must be "
                "[x, y, width, height]"
            )

        x, y, w, h = [
            float(v)
            for v in MANUAL_BBOX
        ]

        if w <= 0 or h <= 0:

            raise ValueError(
                "MANUAL_BBOX width/height "
                "must be positive."
            )

        print()
        print("=" * 72)
        print("MANUAL PERSON BBOX")
        print("=" * 72)

        print(
            "BBox:",
            MANUAL_BBOX
        )

        return [
            x,
            y,
            w,
            h
        ]

    # --------------------------------------------------------
    # 2. COCO
    # --------------------------------------------------------

    bbox = find_coco_bbox(
        image_path,
        ANNOTATION_FILE
    )

    if bbox is not None:

        print()
        print(
            "Using COCO annotation bbox."
        )

        return bbox

    # --------------------------------------------------------
    # 3. YOLO
    # --------------------------------------------------------

    print()
    print(
        "Image is not available in COCO."
    )

    print(
        "Using YOLO person detection."
    )

    return detect_person_bbox(
        image_path
    )


# ============================================================
# SOFT ARGMAX 2D
#
# Matches the coordinate representation used by
# Experiment 10:
#
# coordinates are normalized to [0,1]
# ============================================================

def soft_argmax_2d(
    heatmaps,
    beta=50.0
):

    batch_size, num_keypoints, height, width = (
        heatmaps.shape
    )

    heatmaps_flat = heatmaps.reshape(
        batch_size,
        num_keypoints,
        -1
    )

    probabilities = F.softmax(
        heatmaps_flat * beta,
        dim=-1
    )

    ys = torch.linspace(
        0.0,
        1.0,
        height,
        device=heatmaps.device,
        dtype=heatmaps.dtype
    )

    xs = torch.linspace(
        0.0,
        1.0,
        width,
        device=heatmaps.device,
        dtype=heatmaps.dtype
    )

    yy, xx = torch.meshgrid(
        ys,
        xs,
        indexing="ij"
    )

    xx = xx.reshape(
        1,
        1,
        -1
    )

    yy = yy.reshape(
        1,
        1,
        -1
    )

    x = (
        probabilities
        *
        xx
    ).sum(
        dim=-1
    )

    y = (
        probabilities
        *
        yy
    ).sum(
        dim=-1
    )

    return torch.stack(
        [
            x,
            y
        ],
        dim=-1
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print()
    print("=" * 72)
    print("LOADING EXPERIMENT 10 MODEL")
    print("=" * 72)

    print()
    print(
        "Model:"
    )

    print(
        MODEL_PATH
    )

    if not os.path.isfile(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            "\nModel checkpoint not found:\n"
            + MODEL_PATH
        )

    model = UNetPose(
        NUM_KEYPOINTS
    ).to(
        DEVICE
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Full checkpoint
    # --------------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and
        "model_state_dict"
        in checkpoint
    ):

        print()
        print(
            "Full training checkpoint detected."
        )

        print(
            "Checkpoint epoch:",
            checkpoint.get(
                "epoch",
                "unknown"
            )
        )

        print(
            "Checkpoint metrics:",
            checkpoint.get(
                "metrics",
                "unknown"
            )
        )

        state_dict = checkpoint[
            "model_state_dict"
        ]

    # --------------------------------------------------------
    # Raw state dictionary
    # --------------------------------------------------------

    else:

        print()
        print(
            "Raw model state dictionary detected."
        )

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.eval()

    print()
    print(
        "Model loaded successfully!"
    )

    return model


# ============================================================
# PREPROCESS PERSON CROP
#
# THIS MATCHES YOUR DATASET:
#
# image.crop(...)
# resize(384,384)
# RGB
# /255
# ImageNet normalization
# ============================================================

def preprocess_person_crop(
    image,
    bbox
):

    image_h, image_w = (
        image.shape[:2]
    )

    # --------------------------------------------------------
    # xywh
    # --------------------------------------------------------

    x, y, w, h = [
        float(v)
        for v in bbox
    ]

    # --------------------------------------------------------
    # xyxy
    # --------------------------------------------------------

    bx1 = max(
        0.0,
        min(
            x,
            image_w - 1
        )
    )

    by1 = max(
        0.0,
        min(
            y,
            image_h - 1
        )
    )

    bx2 = max(
        bx1 + 1.0,
        min(
            x + w,
            float(image_w)
        )
    )

    by2 = max(
        by1 + 1.0,
        min(
            y + h,
            float(image_h)
        )
    )

    # --------------------------------------------------------
    # EXACT TRAINING SQUARE BBOX
    # --------------------------------------------------------

    cx1, cy1, cx2, cy2 = (
        make_square_bbox(
            (
                bx1,
                by1,
                bx2,
                by2
            ),
            image_w,
            image_h,
            BBOX_SCALE
        )
    )

    # --------------------------------------------------------
    # EXACT TRAINING ROUNDING
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    actual_x1 = max(
        0,
        min(
            actual_x1,
            image_w - 1
        )
    )

    actual_y1 = max(
        0,
        min(
            actual_y1,
            image_h - 1
        )
    )

    actual_x2 = max(
        actual_x1 + 1,
        min(
            actual_x2,
            image_w
        )
    )

    actual_y2 = max(
        actual_y1 + 1,
        min(
            actual_y2,
            image_h
        )
    )

    # --------------------------------------------------------
    # Crop
    #
    # OpenCV slicing is equivalent to the PIL crop
    # used in your Dataset for this integer box.
    # --------------------------------------------------------

    crop = image[
        actual_y1:actual_y2,
        actual_x1:actual_x2
    ]

    if crop.size == 0:

        raise RuntimeError(
            "Person crop is empty."
        )

    crop_h, crop_w = (
        crop.shape[:2]
    )

    # --------------------------------------------------------
    # Resize EXACTLY to training size
    # --------------------------------------------------------

    crop = cv2.resize(
        crop,
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        interpolation=cv2.INTER_LINEAR
    )

    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    crop = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # /255 EXACTLY AS TRAINING
    # --------------------------------------------------------

    crop = (
        crop.astype(
            np.float32
        )
        /
        255.0
    )

    # --------------------------------------------------------
    # ImageNet normalization
    # EXACTLY AS TRAINING
    # --------------------------------------------------------

    crop = (
        crop - MEAN
    ) / STD

    # --------------------------------------------------------
    # HWC -> CHW
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        crop.transpose(
            2,
            0,
            1
        )
    ).float()

    tensor = tensor.unsqueeze(
        0
    ).to(
        DEVICE
    )

    return (
        tensor,
        actual_x1,
        actual_y1,
        crop_w,
        crop_h
    )


# ============================================================
# COORDINATES TO ORIGINAL IMAGE
#
# Dataset training:
#
# norm_x = kp_x / crop_w
# norm_y = kp_y / crop_h
#
# Therefore inference:
#
# norm_x * crop_w + crop_x1
# norm_y * crop_h + crop_y1
# ============================================================

def coordinates_to_original(
    coordinates,
    crop_x1,
    crop_y1,
    crop_w,
    crop_h,
    image_w,
    image_h
):

    coordinates = coordinates.copy()

    # --------------------------------------------------------
    # normalized -> crop
    # --------------------------------------------------------

    coordinates[:, 0] *= float(
        crop_w
    )

    coordinates[:, 1] *= float(
        crop_h
    )

    # --------------------------------------------------------
    # crop -> original
    # --------------------------------------------------------

    coordinates[:, 0] += float(
        crop_x1
    )

    coordinates[:, 1] += float(
        crop_y1
    )

    # --------------------------------------------------------
    # Clip
    # --------------------------------------------------------

    coordinates[:, 0] = np.clip(
        coordinates[:, 0],
        0,
        image_w - 1
    )

    coordinates[:, 1] = np.clip(
        coordinates[:, 1],
        0,
        image_h - 1
    )

    return coordinates


# ============================================================
# DRAW BBOX
# ============================================================

def draw_bbox(
    image,
    bbox
):

    output = image.copy()

    x, y, w, h = [
        float(v)
        for v in bbox
    ]

    x1 = int(
        round(x)
    )

    y1 = int(
        round(y)
    )

    x2 = int(
        round(x + w)
    )

    y2 = int(
        round(y + h)
    )

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2
    )

    return output


# ============================================================
# DRAW POSE
# ============================================================

def draw_pose(
    image,
    coordinates,
    visibility
):

    output = image.copy()

    points = []

    # --------------------------------------------------------
    # Convert to integer points
    # --------------------------------------------------------

    for x, y in coordinates:

        points.append(
            (
                int(round(x)),
                int(round(y))
            )
        )

    # --------------------------------------------------------
    # Skeleton
    # --------------------------------------------------------

    for a, b in SKELETON:

        if (
            visibility[a]
            >= VISIBILITY_THRESHOLD
            and
            visibility[b]
            >= VISIBILITY_THRESHOLD
        ):

            cv2.line(
                output,
                points[a],
                points[b],
                (0, 255, 0),
                3,
                cv2.LINE_AA
            )

    # --------------------------------------------------------
    # Keypoints
    # --------------------------------------------------------

    for k, (x, y) in enumerate(
        points
    ):

        if (
            visibility[k]
            < VISIBILITY_THRESHOLD
        ):

            continue

        cv2.circle(
            output,
            (x, y),
            6,
            (0, 0, 255),
            -1,
            cv2.LINE_AA
        )

        cv2.putText(
            output,
            str(k),
            (
                x + 7,
                y - 7
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print(
        "COCO POSE INFERENCE - "
        "EXPERIMENT 10"
    )
    print(
        "EXACT TRAINING PREPROCESSING"
    )
    print("=" * 72)

    # ========================================================
    # DEVICE
    # ========================================================

    print()
    print(
        "Device:",
        DEVICE
    )

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA available: True"
        )

    else:

        print(
            "CUDA available: False"
        )

    # ========================================================
    # CHECK BASE DIRECTORY
    # ========================================================

    print()
    print(
        "Base directory:"
    )

    print(
        BASE_DIR
    )

    if not os.path.isdir(
        BASE_DIR
    ):

        raise FileNotFoundError(
            "\nBase directory not found:\n"
            + BASE_DIR
        )

    # ========================================================
    # INPUT IMAGE
    # ========================================================

    print()
    print("=" * 72)
    print("INPUT IMAGE")
    print("=" * 72)

    print()
    print(
        "Image:",
        IMAGE_PATH
    )

    if not os.path.isfile(
        IMAGE_PATH
    ):

        raise FileNotFoundError(
            "\nInput image not found:\n"
            + IMAGE_PATH
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = load_model()

    # ========================================================
    # READ IMAGE
    # ========================================================

    image = cv2.imread(
        IMAGE_PATH
    )

    if image is None:

        raise RuntimeError(
            "\nOpenCV could not read image:\n"
            + IMAGE_PATH
        )

    original = image.copy()

    original_h, original_w = (
        image.shape[:2]
    )

    print()
    print(
        "Original size:",
        original_w,
        "x",
        original_h
    )

    # ========================================================
    # GET PERSON BBOX
    # ========================================================

    bbox = get_bbox(
        IMAGE_PATH,
        original_w,
        original_h
    )

    # ========================================================
    # PERSON CROP
    # ========================================================

    print()
    print("=" * 72)
    print("PERSON CROP PREPROCESSING")
    print("=" * 72)

    (
        tensor,
        crop_x1,
        crop_y1,
        crop_w,
        crop_h
    ) = preprocess_person_crop(
        image,
        bbox
    )

    print()
    print(
        "Original person bbox:"
    )

    print(
        [
            round(
                float(v),
                2
            )
            for v in bbox
        ]
    )

    print()
    print(
        "Final training crop:"
    )

    print(
        f"x1={crop_x1}, "
        f"y1={crop_y1}, "
        f"width={crop_w}, "
        f"height={crop_h}"
    )

    print()
    print(
        "Model input:",
        tensor.shape
    )

    # ========================================================
    # MODEL INFERENCE
    #
    # IMPORTANT:
    # model is defined above.
    # This MUST stay inside main().
    # ========================================================

    print()
    print("=" * 72)
    print("RUNNING INFERENCE")
    print("=" * 72)

    with torch.no_grad():

        pred_heatmaps, pred_visibility = (
            model(tensor)
        )

        coordinates = soft_argmax_2d(
            pred_heatmaps,
            beta=SOFTARGMAX_BETA
        )

        visibility = torch.sigmoid(
            pred_visibility
        )

    # ========================================================
    # OUTPUT SHAPES
    # ========================================================

    print()
    print(
        "Heatmap output:",
        pred_heatmaps.shape
    )

    print(
        "Visibility output:",
        pred_visibility.shape
    )

    print(
        "Coordinates:",
        coordinates.shape
    )

    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    coordinates = (
        coordinates[0]
        .detach()
        .cpu()
        .numpy()
    )

    visibility = (
        visibility[0]
        .detach()
        .cpu()
        .numpy()
    )

    # ========================================================
    # CONVERT TO ORIGINAL IMAGE
    # ========================================================

    original_coordinates = (
        coordinates_to_original(
            coordinates,
            crop_x1,
            crop_y1,
            crop_w,
            crop_h,
            original_w,
            original_h
        )
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 72)
    print(
        "PREDICTED KEYPOINTS - "
        "ORIGINAL IMAGE COORDINATES"
    )
    print("=" * 72)

    print()

    print(
        f"{'ID':<4}"
        f"{'KEYPOINT':<18}"
        f"{'X':>10}"
        f"{'Y':>10}"
        f"{'VIS':>10}"
    )

    print(
        "-" * 60
    )

    for k in range(
        NUM_KEYPOINTS
    ):

        x, y = (
            original_coordinates[k]
        )

        print(
            f"{k:<4}"
            f"{KEYPOINT_NAMES[k]:<18}"
            f"{x:>10.2f}"
            f"{y:>10.2f}"
            f"{visibility[k]:>10.3f}"
        )

    # ========================================================
    # DRAW
    # ========================================================

    output = draw_pose(
        original,
        original_coordinates,
        visibility
    )

    # ========================================================
    # SAVE
    # ========================================================

    success = cv2.imwrite(
        OUTPUT_PATH,
        output
    )

    if not success:

        raise RuntimeError(
            "\nFailed to save output image:\n"
            + OUTPUT_PATH
        )

    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print("=" * 72)
    print("INFERENCE COMPLETED")
    print("=" * 72)

    print()
    print(
        "Input:"
    )

    print(
        IMAGE_PATH
    )

    print()
    print(
        "Output:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print(
        "Image size:",
        f"{original_w} x {original_h}"
    )

    print(
        "BBOX SCALE:",
        BBOX_SCALE
    )

    print(
        "IMAGE SIZE:",
        IMAGE_SIZE
    )

    print(
        "HEATMAP SIZE:",
        HEATMAP_SIZE
    )

    print(
        "SOFTARGMAX BETA:",
        SOFTARGMAX_BETA
    )

    print(
        "VISIBILITY THRESHOLD:",
        VISIBILITY_THRESHOLD
    )

    print()
    print(
        "Done!"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
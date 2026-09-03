import os


# ============================================================
# PATHS
# ============================================================

IMAGE_DIR = "/content/drive/MyDrive/fiftyone/coco-2017/train/data"

ANNOTATION_FILE = (
    "/content/drive/MyDrive/fiftyone/coco-2017/raw/"
    "person_keypoints_train2017.json"
)

OUTPUT_DIR = (
    "/content/drive/MyDrive/ai dataset/"
    "checkpoints/experiment11"
)

EVAL_DIR = (
    "/content/drive/MyDrive/ai dataset/"
    "evaluation_results/experiment11"
)


# ============================================================
# IMAGE / HEATMAP
# ============================================================

IMAGE_SIZE = 384
HEATMAP_SIZE = 96

NUM_KEYPOINTS = 17


# ============================================================
# TRAINING
# ============================================================

BATCH_SIZE = 32

EPOCHS = 30

LEARNING_RATE = 2e-4

WEIGHT_DECAY = 1e-4

TRAIN_RATIO = 0.80

SEED = 42


# ============================================================
# HEATMAP
# ============================================================

SIGMA = 2.5

HEATMAP_WEIGHT = 1.0

COORD_WEIGHT = 2.0

VIS_WEIGHT = 0.25

COORD_HUBER_BETA = 0.03


# ============================================================
# PERSON CROP
# ============================================================

BBOX_SCALE = 1.20


# ============================================================
# DATA FILTERING
# ============================================================

MIN_VISIBLE_KEYPOINTS = 5

MAX_SAMPLES = None


# ============================================================
# DATALOADER
# ============================================================

NUM_WORKERS = 2

PIN_MEMORY = True

PERSISTENT_WORKERS = True

EVAL_BATCH_SIZE = 64


# ============================================================
# CHECKPOINTS
# ============================================================

SAVE_LAST_EVERY_EPOCH = True


# ============================================================
# SOFTARGMAX
# ============================================================

SOFTARGMAX_BETA = 50.0


# ============================================================
# TRAINING AUGMENTATION
# ============================================================

HORIZONTAL_FLIP_PROB = 0.5

COLOR_JITTER_PROB = 0.30

BRIGHTNESS_RANGE = 0.20

CONTRAST_RANGE = 0.20

SATURATION_RANGE = 0.20

HUE_RANGE = 0.05

SCALE_AUG_PROB = 0.30

SCALE_AUG_MIN = 0.90

SCALE_AUG_MAX = 1.10

TRANSLATION_AUG_PROB = 0.30

TRANSLATION_AUG_MAX = 0.08


# ============================================================
# LR SCHEDULER
# ============================================================

MIN_LEARNING_RATE = 2e-6


# ============================================================
# COCO KEYPOINT NAMES
# ============================================================

COCO_NAMES = [
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
    "right_ankle",
]


# ============================================================
# COCO FLIP PAIRS
# ============================================================

FLIP_PAIRS = [
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
]
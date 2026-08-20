import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["keypoints"],
    classes=["person"],
    max_samples=3000,
)

print("Dataset loaded!")
print("Number of samples:", len(dataset))

session = fo.launch_app(dataset)
session.wait()
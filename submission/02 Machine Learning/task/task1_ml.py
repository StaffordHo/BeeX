from __future__ import annotations

import argparse
import json
import math
import os
import random
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from segmentation_models_pytorch import UnetPlusPlus
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


CLASS_NAMES = ("background", "anode", "member")
TARGET_CATEGORY_TO_TRAIN_ID = {
    "anode": 1,
    "member": 2,
}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    file_name: str
    width: int
    height: int
    annotations: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class AugmentationConfig:
    enabled: bool
    hflip_prob: float
    affine_prob: float
    max_rotate_deg: float
    scale_jitter: float
    translate_jitter: float
    intensity_prob: float
    noise_prob: float
    blur_prob: float


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train/evaluate UnetPlusPlus for BeeX sonar member/anode segmentation."
    )
    parser.add_argument("--dataset-dir", type=Path, default=script_dir / "interview_dataset")
    parser.add_argument("--zip-path", type=Path, default=script_dir / "interview_dataset.zip")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "outputs")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional dataset cap for quick local runs.")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny one-epoch check without expensive training.")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not load ImageNet encoder weights.")
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--anode-dice-weight", type=float, default=0.25)
    parser.add_argument("--anode-class-weight-multiplier", type=float, default=3.0)
    parser.add_argument("--anode-sample-weight", type=float, default=6.0)
    parser.add_argument("--member-sample-weight", type=float, default=1.0)
    parser.add_argument("--background-sample-weight", type=float, default=0.5)
    parser.add_argument("--augment", action="store_true", help="Enable conservative train-only sonar augmentation.")
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--affine-prob", type=float, default=0.35)
    parser.add_argument("--max-rotate-deg", type=float, default=3.0)
    parser.add_argument("--scale-jitter", type=float, default=0.04)
    parser.add_argument("--translate-jitter", type=float, default=0.03)
    parser.add_argument("--intensity-prob", type=float, default=0.5)
    parser.add_argument("--noise-prob", type=float, default=0.3)
    parser.add_argument("--blur-prob", type=float, default=0.2)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint to evaluate/resume from.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and evaluate a checkpoint.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def validate_probability(name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")


def build_augmentation_config(args: argparse.Namespace) -> AugmentationConfig | None:
    if not args.augment:
        return None

    for name in ("hflip_prob", "affine_prob", "intensity_prob", "noise_prob", "blur_prob"):
        validate_probability(f"--{name.replace('_', '-')}", float(getattr(args, name)))
    if args.max_rotate_deg < 0.0:
        raise ValueError("--max-rotate-deg must be non-negative")
    if args.scale_jitter < 0.0 or args.scale_jitter > 0.5:
        raise ValueError("--scale-jitter must be between 0.0 and 0.5")
    if args.translate_jitter < 0.0 or args.translate_jitter > 0.5:
        raise ValueError("--translate-jitter must be between 0.0 and 0.5")

    return AugmentationConfig(
        enabled=True,
        hflip_prob=args.hflip_prob,
        affine_prob=args.affine_prob,
        max_rotate_deg=args.max_rotate_deg,
        scale_jitter=args.scale_jitter,
        translate_jitter=args.translate_jitter,
        intensity_prob=args.intensity_prob,
        noise_prob=args.noise_prob,
        blur_prob=args.blur_prob,
    )


def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"Refusing to extract unsafe zip path: {member.filename}")
        archive.extractall(destination)


def ensure_dataset(dataset_dir: Path, zip_path: Path) -> Path:
    annotation_file = dataset_dir / "annotations" / "instances_default.json"
    if annotation_file.exists():
        return dataset_dir

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_dir} and zip archive not found at {zip_path}"
        )

    print(f"Extracting {zip_path} to {dataset_dir.parent}")
    safe_extract_zip(zip_path, dataset_dir.parent)
    if not annotation_file.exists():
        raise FileNotFoundError(f"Expected annotation file after extraction: {annotation_file}")
    return dataset_dir


def load_coco_records(dataset_dir: Path) -> tuple[list[ImageRecord], dict[int, int]]:
    annotation_file = dataset_dir / "annotations" / "instances_default.json"
    with annotation_file.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    category_id_to_train_id: dict[int, int] = {}
    for category in coco["categories"]:
        name = str(category["name"]).lower()
        if name in TARGET_CATEGORY_TO_TRAIN_ID:
            category_id_to_train_id[int(category["id"])] = TARGET_CATEGORY_TO_TRAIN_ID[name]

    if set(category_id_to_train_id.values()) != {1, 2}:
        raise RuntimeError("Expected COCO categories named 'anode' and 'member'.")

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        if int(annotation["category_id"]) in category_id_to_train_id:
            annotations_by_image[int(annotation["image_id"])].append(annotation)

    records: list[ImageRecord] = []
    for image in sorted(coco["images"], key=lambda item: int(item["id"])):
        image_id = int(image["id"])
        records.append(
            ImageRecord(
                image_id=image_id,
                file_name=str(image["file_name"]),
                width=int(image["width"]),
                height=int(image["height"]),
                annotations=tuple(annotations_by_image.get(image_id, ())),
            )
        )

    return records, category_id_to_train_id


def draw_polygon_mask(
    record: ImageRecord,
    category_id_to_train_id: dict[int, int],
) -> np.ndarray:
    mask = np.zeros((record.height, record.width), dtype=np.uint8)

    # Draw member first and anode second so the rarer anode label is preserved if polygons overlap.
    annotations = sorted(
        record.annotations,
        key=lambda annotation: category_id_to_train_id[int(annotation["category_id"])],
        reverse=True,
    )
    for annotation in annotations:
        train_id = category_id_to_train_id[int(annotation["category_id"])]
        segmentation = annotation.get("segmentation", [])
        if not isinstance(segmentation, list):
            raise ValueError("Only polygon COCO segmentations are supported by this lightweight parser.")

        for polygon in segmentation:
            if len(polygon) < 6:
                continue
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            points[:, 0] = np.clip(points[:, 0], 0, record.width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, record.height - 1)
            cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], int(train_id))

    return mask


def letterbox_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    scale = min(output_size / width, output_size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    canvas_image = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    canvas_mask = np.zeros((output_size, output_size), dtype=np.uint8)
    top = (output_size - new_height) // 2
    left = (output_size - new_width) // 2
    canvas_image[top : top + new_height, left : left + new_width] = resized_image
    canvas_mask[top : top + new_height, left : left + new_width] = resized_mask
    return canvas_image, canvas_mask


def anode_would_be_erased(before_mask: np.ndarray, after_mask: np.ndarray) -> bool:
    return bool(np.any(before_mask == 1) and not np.any(after_mask == 1))


def foreground_would_drop_too_much(before_mask: np.ndarray, after_mask: np.ndarray) -> bool:
    before_count = int(np.count_nonzero(before_mask))
    if before_count == 0:
        return False
    after_count = int(np.count_nonzero(after_mask))
    return after_count < max(1, int(math.ceil(0.5 * before_count)))


def apply_affine_augmentation(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    config: AugmentationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if random.random() >= config.affine_prob:
        return image_rgb, mask

    height, width = mask.shape[:2]
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    angle = random.uniform(-config.max_rotate_deg, config.max_rotate_deg)
    scale = random.uniform(1.0 - config.scale_jitter, 1.0 + config.scale_jitter)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += random.uniform(-config.translate_jitter, config.translate_jitter) * width
    matrix[1, 2] += random.uniform(-config.translate_jitter, config.translate_jitter) * height

    warped_image = cv2.warpAffine(
        image_rgb,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_mask = cv2.warpAffine(
        mask,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Anodes are tiny in this dataset; reject a geometry sample if it removes them completely.
    if anode_would_be_erased(mask, warped_mask) or foreground_would_drop_too_much(mask, warped_mask):
        return image_rgb, mask
    return warped_image, warped_mask.astype(np.uint8, copy=False)


def apply_intensity_augmentation(image_rgb: np.ndarray, config: AugmentationConfig) -> np.ndarray:
    augmented = image_rgb

    if random.random() < config.intensity_prob:
        image_float = augmented.astype(np.float32) / 255.0
        contrast = random.uniform(0.85, 1.15)
        brightness = random.uniform(-0.08, 0.08)
        gamma = random.uniform(0.85, 1.15)
        image_float = np.clip((image_float - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
        image_float = np.power(image_float, gamma)
        augmented = np.clip(image_float * 255.0, 0.0, 255.0).astype(np.uint8)

    if random.random() < config.noise_prob:
        noise_sigma = random.uniform(2.0, 8.0)
        noise = np.random.normal(0.0, noise_sigma, size=augmented.shape)
        augmented = np.clip(augmented.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)

    if random.random() < config.blur_prob:
        augmented = cv2.GaussianBlur(augmented, (3, 3), sigmaX=0.0)

    return augmented


def apply_train_augmentation(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    config: AugmentationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < config.hflip_prob:
        image_rgb = cv2.flip(image_rgb, 1)
        mask = cv2.flip(mask, 1)

    image_rgb, mask = apply_affine_augmentation(image_rgb, mask, config)
    image_rgb = apply_intensity_augmentation(image_rgb, config)
    return np.ascontiguousarray(image_rgb), np.ascontiguousarray(mask.astype(np.uint8, copy=False))


def image_stratification_label(record: ImageRecord, category_id_to_train_id: dict[int, int]) -> str:
    train_ids = {category_id_to_train_id[int(annotation["category_id"])] for annotation in record.annotations}
    if 1 in train_ids:
        return "anode"
    if 2 in train_ids:
        return "member"
    return "background"


def describe_dataset(records: list[ImageRecord], category_id_to_train_id: dict[int, int]) -> dict[str, Any]:
    annotation_counts = Counter()
    image_presence = Counter()
    for record in records:
        present = set()
        for annotation in record.annotations:
            train_id = category_id_to_train_id[int(annotation["category_id"])]
            annotation_counts[CLASS_NAMES[train_id]] += 1
            present.add(CLASS_NAMES[train_id])
        image_presence["+".join(sorted(present)) if present else "background"] += 1

    return {
        "num_images": len(records),
        "num_annotations": int(sum(annotation_counts.values())),
        "annotation_counts": dict(annotation_counts),
        "image_presence": dict(image_presence),
    }


def describe_split(records: list[ImageRecord], category_id_to_train_id: dict[int, int]) -> dict[str, Any]:
    labels = Counter(image_stratification_label(record, category_id_to_train_id) for record in records)
    return {"num_images": len(records), "image_presence": dict(labels)}


class SonarSegmentationDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        records: list[ImageRecord],
        category_id_to_train_id: dict[int, int],
        image_size: int,
        augmentation: AugmentationConfig | None = None,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.records = records
        self.category_id_to_train_id = category_id_to_train_id
        self.image_size = image_size
        self.augmentation = augmentation
        self.images_dir = dataset_dir / "images"

    def __len__(self) -> int:
        return len(self.records)

    def load_mask(self, index: int) -> np.ndarray:
        record = self.records[index]
        mask = draw_polygon_mask(record, self.category_id_to_train_id)
        dummy_image = np.zeros((record.height, record.width, 3), dtype=np.uint8)
        _, resized_mask = letterbox_image_and_mask(dummy_image, mask, self.image_size)
        return resized_mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image_path = self.images_dir / record.file_name
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = draw_polygon_mask(record, self.category_id_to_train_id)
        image_rgb, mask = letterbox_image_and_mask(image_rgb, mask, self.image_size)
        if self.augmentation is not None and self.augmentation.enabled:
            image_rgb, mask = apply_train_augmentation(image_rgb, mask, self.augmentation)

        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float().div(255.0)
        image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD
        mask_tensor = torch.from_numpy(mask.astype(np.int64))
        return image_tensor, mask_tensor


def split_records(
    records: list[ImageRecord],
    category_id_to_train_id: dict[int, int],
    val_size: float,
    test_size: float,
    seed: int,
    max_samples: int | None,
) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord]]:
    if val_size < 0.0 or test_size < 0.0 or val_size + test_size >= 1.0:
        raise ValueError("--val-size and --test-size must be non-negative and sum to less than 1.0")

    rng = random.Random(seed)
    if max_samples is not None:
        grouped_records: dict[str, list[ImageRecord]] = defaultdict(list)
        for record in records:
            grouped_records[image_stratification_label(record, category_id_to_train_id)].append(record)
        for group in grouped_records.values():
            rng.shuffle(group)

        selected: list[ImageRecord] = []
        label_order = ["anode", "member", "background"]
        while len(selected) < max_samples and any(grouped_records.values()):
            for label in label_order:
                if grouped_records[label]:
                    selected.append(grouped_records[label].pop())
                    if len(selected) >= max_samples:
                        break
        records = selected

    stratify_labels = [image_stratification_label(record, category_id_to_train_id) for record in records]
    label_counts = Counter(stratify_labels)
    train_records: list[ImageRecord] = []
    valid_records: list[ImageRecord] = []
    test_records: list[ImageRecord] = []

    if label_counts and min(label_counts.values()) >= 3:
        grouped_records: dict[str, list[ImageRecord]] = defaultdict(list)
        for record, label in zip(records, stratify_labels):
            grouped_records[label].append(record)

        for group in grouped_records.values():
            rng.shuffle(group)
            valid_count = max(1, int(round(len(group) * val_size)))
            test_count = max(1, int(round(len(group) * test_size))) if test_size > 0.0 else 0
            test_count = min(test_count, len(group) - 2)
            valid_count = min(valid_count, len(group) - test_count - 1)
            test_records.extend(group[:test_count])
            valid_records.extend(group[test_count : test_count + valid_count])
            train_records.extend(group[test_count + valid_count :])
    else:
        shuffled = list(records)
        rng.shuffle(shuffled)
        valid_count = max(1, int(round(len(shuffled) * val_size)))
        test_count = max(1, int(round(len(shuffled) * test_size))) if test_size > 0.0 else 0
        test_count = min(test_count, len(shuffled) - 2)
        valid_count = min(valid_count, len(shuffled) - test_count - 1)
        test_records = shuffled[:test_count]
        valid_records = shuffled[test_count : test_count + valid_count]
        train_records = shuffled[test_count + valid_count :]

    rng.shuffle(train_records)
    rng.shuffle(valid_records)
    rng.shuffle(test_records)
    return train_records, valid_records, test_records


def make_sample_weights(
    records: list[ImageRecord],
    category_id_to_train_id: dict[int, int],
    anode_weight: float,
    member_weight: float,
    background_weight: float,
) -> torch.Tensor:
    label_to_weight = {
        "anode": anode_weight,
        "member": member_weight,
        "background": background_weight,
    }
    weights = [label_to_weight[image_stratification_label(record, category_id_to_train_id)] for record in records]
    return torch.as_tensor(weights, dtype=torch.double)


def compute_class_weights(dataset: SonarSegmentationDataset) -> torch.Tensor:
    counts = torch.zeros(len(CLASS_NAMES), dtype=torch.float64)
    for index in tqdm(range(len(dataset)), desc="Counting train pixels"):
        mask = dataset.load_mask(index)
        values = torch.bincount(torch.from_numpy(mask.reshape(-1)).long(), minlength=len(CLASS_NAMES))
        counts += values.double()

    frequencies = counts / counts.sum().clamp_min(1.0)
    weights = 1.0 / torch.log(1.02 + frequencies)
    weights = weights / weights.mean().clamp_min(1e-6)
    weights = torch.clamp(weights, min=0.25, max=6.0).float()
    return weights


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, include_background: bool = False) -> torch.Tensor:
    num_classes = logits.shape[1]
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    intersection = torch.sum(probabilities * one_hot, dims)
    cardinality = torch.sum(probabilities + one_hot, dims)
    dice = (2.0 * intersection + 1.0) / (cardinality + 1.0)

    present_classes = torch.sum(one_hot, dims) > 0
    if not include_background:
        present_classes[0] = False
    if not torch.any(present_classes):
        return logits.sum() * 0.0
    return 1.0 - dice[present_classes].mean()


def binary_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    binary_targets = targets.float()
    if torch.sum(binary_targets) == 0:
        return logits.sum() * 0.0
    dims = (0, 1, 2)
    intersection = torch.sum(probabilities * binary_targets, dims)
    cardinality = torch.sum(probabilities + binary_targets, dims)
    dice = (2.0 * intersection + 1.0) / (cardinality + 1.0)
    return 1.0 - dice


def segmentation_logits(model_output: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]) -> torch.Tensor:
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    return model_output


def update_confusion_matrix(confusion: torch.Tensor, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    num_classes = confusion.shape[0]
    predictions = predictions.view(-1).to(torch.int64)
    targets = targets.view(-1).to(torch.int64)
    valid = (targets >= 0) & (targets < num_classes)
    indices = num_classes * targets[valid] + predictions[valid]
    confusion += torch.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes).cpu()
    return confusion


def metrics_from_confusion(confusion: torch.Tensor) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for class_id in (1, 2):
        tp = confusion[class_id, class_id].item()
        fp = confusion[:, class_id].sum().item() - tp
        fn = confusion[class_id, :].sum().item() - tp

        iou_denominator = tp + fp + fn
        f1_denominator = 2 * tp + fp + fn
        results[CLASS_NAMES[class_id]] = {
            "iou": float(tp / iou_denominator) if iou_denominator > 0 else 0.0,
            "f1": float((2 * tp) / f1_denominator) if f1_denominator > 0 else 0.0,
        }
    return results


def pixel_counts_from_confusion(confusion: torch.Tensor) -> dict[str, dict[str, int]]:
    target_pixels = confusion.sum(dim=1)
    predicted_pixels = confusion.sum(dim=0)
    return {
        "target_pixels": {name: int(target_pixels[index].item()) for index, name in enumerate(CLASS_NAMES)},
        "predicted_pixels": {name: int(predicted_pixels[index].item()) for index, name in enumerate(CLASS_NAMES)},
    }


def make_model(no_pretrained: bool) -> nn.Module:
    encoder_weights = None if no_pretrained else "imagenet"
    return UnetPlusPlus(
        "resnet34",
        encoder_weights=encoder_weights,
        classes=len(CLASS_NAMES),
        activation=None,
    )


def run_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    ce_loss: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    dice_weight: float,
    anode_dice_weight: float,
    phase: str,
) -> tuple[float, dict[str, dict[str, float]], dict[str, dict[str, int]]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_batches = 0
    confusion = torch.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=torch.int64)

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        progress = tqdm(dataloader, desc=phase)
        for images, targets in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = segmentation_logits(model(images))
            loss = ce_loss(logits, targets) + dice_weight * soft_dice_loss(logits, targets)
            if anode_dice_weight > 0.0:
                loss = loss + anode_dice_weight * binary_dice_loss(logits[:, 1], targets == 1)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            predictions = torch.argmax(logits.detach(), dim=1)
            update_confusion_matrix(confusion, predictions.cpu(), targets.cpu())
            total_loss += float(loss.detach().cpu().item())
            total_batches += 1
            progress.set_postfix(loss=f"{total_loss / max(total_batches, 1):.4f}")

    return total_loss / max(total_batches, 1), metrics_from_confusion(confusion), pixel_counts_from_confusion(confusion)


def save_loss_curve(history: list[dict[str, Any]], output_path: Path) -> None:
    try:
        if "MPLCONFIGDIR" not in os.environ:
            matplotlib_cache_dir = Path("/tmp") / "matplotlib-cache"
            matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(matplotlib_cache_dir)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"Skipping loss curve because matplotlib is unavailable: {exc}")
        return

    epochs = [entry["epoch"] for entry in history]
    train_losses = [entry["train_loss"] for entry in history]
    valid_losses = [entry["valid_loss"] for entry in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="train")
    plt.plot(epochs, valid_losses, marker="o", label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Segmentation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def write_metrics_text(metrics: dict[str, Any], output_path: Path) -> None:
    lines = [
        "BeeX Task 2 Metrics",
        "",
        f"Best validation epoch: {metrics.get('best_epoch')}",
        f"Best validation mean target IoU: {metrics.get('best_mean_iou', 0.0):.4f}",
        "",
        "Validation:",
    ]
    for class_name in ("anode", "member"):
        class_metrics = metrics["validation_metrics"][class_name]
        lines.append(f"{class_name}: IoU={class_metrics['iou']:.4f}, F1={class_metrics['f1']:.4f}")
    lines.append("")
    lines.append("Test:")
    for class_name in ("anode", "member"):
        class_metrics = metrics["test_metrics"][class_name]
        lines.append(f"{class_name}: IoU={class_metrics['iou']:.4f}, F1={class_metrics['f1']:.4f}")

    predicted = metrics.get("test_pixel_counts", {}).get("predicted_pixels", {})
    if predicted:
        lines.append("")
        lines.append(f"Test predicted pixels: {predicted}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.max_samples = args.max_samples or 32
        args.num_workers = 0

    set_seed(args.seed)
    augmentation_config = build_augmentation_config(args)
    if augmentation_config is not None:
        print(f"Train augmentation: {augmentation_config}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = ensure_dataset(args.dataset_dir, args.zip_path)
    records, category_id_to_train_id = load_coco_records(dataset_dir)
    dataset_summary = describe_dataset(records, category_id_to_train_id)
    print(json.dumps(dataset_summary, indent=2))

    train_records, valid_records, test_records = split_records(
        records,
        category_id_to_train_id,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    print(
        f"Train images: {len(train_records)} | "
        f"Validation images: {len(valid_records)} | "
        f"Test images: {len(test_records)}"
    )
    split_summary = {
        "train": describe_split(train_records, category_id_to_train_id),
        "validation": describe_split(valid_records, category_id_to_train_id),
        "test": describe_split(test_records, category_id_to_train_id),
    }

    train_dataset = SonarSegmentationDataset(
        dataset_dir,
        train_records,
        category_id_to_train_id,
        args.image_size,
        augmentation=augmentation_config,
    )
    valid_dataset = SonarSegmentationDataset(dataset_dir, valid_records, category_id_to_train_id, args.image_size)
    test_dataset = SonarSegmentationDataset(dataset_dir, test_records, category_id_to_train_id, args.image_size)
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_sampler = WeightedRandomSampler(
        make_sample_weights(
            train_records,
            category_id_to_train_id,
            anode_weight=args.anode_sample_weight,
            member_weight=args.member_sample_weight,
            background_weight=args.background_sample_weight,
        ),
        num_samples=len(train_records),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )

    device = torch.device(args.device)
    class_weights = compute_class_weights(train_dataset).to(device)
    class_weights[1] = class_weights[1] * args.anode_class_weight_multiplier
    print("Class weights:", {name: round(float(weight), 4) for name, weight in zip(CLASS_NAMES, class_weights)})

    model = make_model(args.no_pretrained).to(device)
    if args.checkpoint is not None and args.checkpoint.exists():
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, Any]] = []
    best_mean_iou = -math.inf
    best_epoch = 0
    final_metrics: dict[str, dict[str, float]] = {
        "anode": {"iou": 0.0, "f1": 0.0},
        "member": {"iou": 0.0, "f1": 0.0},
    }
    final_pixel_counts: dict[str, dict[str, int]] = {"target_pixels": {}, "predicted_pixels": {}}
    best_validation_metrics: dict[str, dict[str, float]] = {
        "anode": {"iou": 0.0, "f1": 0.0},
        "member": {"iou": 0.0, "f1": 0.0},
    }
    best_validation_pixel_counts: dict[str, dict[str, int]] = {"target_pixels": {}, "predicted_pixels": {}}
    test_metrics: dict[str, dict[str, float]] = {
        "anode": {"iou": 0.0, "f1": 0.0},
        "member": {"iou": 0.0, "f1": 0.0},
    }
    test_pixel_counts: dict[str, dict[str, int]] = {"target_pixels": {}, "predicted_pixels": {}}

    if args.eval_only:
        valid_loss, final_metrics, final_pixel_counts = run_one_epoch(
            model,
            valid_loader,
            ce_loss,
            None,
            device,
            args.dice_weight,
            args.anode_dice_weight,
            phase="Validation",
        )
        history.append(
            {
                "epoch": 0,
                "train_loss": 0.0,
                "valid_loss": valid_loss,
                "metrics": final_metrics,
                "pixel_counts": final_pixel_counts,
            }
        )
        best_validation_metrics = final_metrics
        best_validation_pixel_counts = final_pixel_counts
    else:
        for epoch in range(1, args.epochs + 1):
            train_loss, _, _ = run_one_epoch(
                model,
                train_loader,
                ce_loss,
                optimizer,
                device,
                args.dice_weight,
                args.anode_dice_weight,
                phase=f"Train {epoch}",
            )
            valid_loss, final_metrics, final_pixel_counts = run_one_epoch(
                model,
                valid_loader,
                ce_loss,
                None,
                device,
                args.dice_weight,
                args.anode_dice_weight,
                phase=f"Valid {epoch}",
            )
            mean_iou = float(np.mean([final_metrics["anode"]["iou"], final_metrics["member"]["iou"]]))
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "metrics": final_metrics,
                    "pixel_counts": final_pixel_counts,
                    "mean_target_iou": mean_iou,
                }
            )

            if mean_iou > best_mean_iou:
                best_mean_iou = mean_iou
                best_epoch = epoch
                best_validation_metrics = final_metrics
                best_validation_pixel_counts = final_pixel_counts
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "class_names": CLASS_NAMES,
                        "image_size": args.image_size,
                        "metrics": final_metrics,
                        "pixel_counts": final_pixel_counts,
                    },
                    args.output_dir / "best_model.pt",
                )

    if history and best_mean_iou == -math.inf:
        best_mean_iou = float(np.mean([final_metrics["anode"]["iou"], final_metrics["member"]["iou"]]))
        best_epoch = int(history[-1]["epoch"])
        best_validation_metrics = final_metrics
        best_validation_pixel_counts = final_pixel_counts

    best_checkpoint = args.output_dir / "best_model.pt"
    if best_checkpoint.exists() and not args.eval_only:
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_metrics, test_pixel_counts = run_one_epoch(
        model,
        test_loader,
        ce_loss,
        None,
        device,
        args.dice_weight,
        args.anode_dice_weight,
        phase="Test",
    )

    metrics_payload = {
        "dataset_summary": dataset_summary,
        "split_summary": split_summary,
        "class_names": CLASS_NAMES,
        "best_epoch": best_epoch,
        "best_mean_iou": best_mean_iou,
        "validation_metrics": best_validation_metrics,
        "validation_pixel_counts": best_validation_pixel_counts,
        "last_validation_metrics": final_metrics,
        "last_validation_pixel_counts": final_pixel_counts,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "test_pixel_counts": test_pixel_counts,
        "final_metrics": test_metrics,
        "history": history,
        "augmentation": augmentation_config.__dict__ if augmentation_config is not None else None,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    write_metrics_text(metrics_payload, args.output_dir / "metrics.txt")
    save_loss_curve(history, args.output_dir / "loss_curves.png")

    print(json.dumps({"validation": best_validation_metrics, "test": test_metrics}, indent=2))
    print(f"Wrote artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()

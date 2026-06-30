from __future__ import annotations

import random
import unittest

import numpy as np
import torch

import task1_ml as ml


CATEGORY_TO_TRAIN_ID = {1: 1, 2: 2}


def make_annotation(category_id: int, polygon: list[float]) -> dict[str, object]:
    return {"category_id": category_id, "segmentation": [polygon]}


def make_record(image_id: int, annotations: tuple[dict[str, object], ...]) -> ml.ImageRecord:
    return ml.ImageRecord(
        image_id=image_id,
        file_name=f"synthetic_{image_id}.png",
        width=10,
        height=10,
        annotations=annotations,
    )


class Task1MlDataPipelineTests(unittest.TestCase):
    def test_polygon_mask_draws_anode_over_member(self) -> None:
        member = make_annotation(2, [1, 1, 8, 1, 8, 8, 1, 8])
        anode = make_annotation(1, [4, 4, 7, 4, 7, 7, 4, 7])
        record = make_record(1, (anode, member))

        mask = ml.draw_polygon_mask(record, CATEGORY_TO_TRAIN_ID)

        self.assertEqual(mask[2, 2], 2)
        self.assertEqual(mask[5, 5], 1)
        self.assertEqual(set(np.unique(mask)).issubset({0, 1, 2}), True)

    def test_letterbox_keeps_class_indices_with_nearest_neighbor_masks(self) -> None:
        image = np.zeros((2, 4, 3), dtype=np.uint8)
        mask = np.array([[0, 1, 2, 2], [0, 1, 2, 2]], dtype=np.uint8)

        resized_image, resized_mask = ml.letterbox_image_and_mask(image, mask, output_size=8)

        self.assertEqual(resized_image.shape, (8, 8, 3))
        self.assertEqual(resized_mask.shape, (8, 8))
        self.assertEqual(set(np.unique(resized_mask)).issubset({0, 1, 2}), True)
        self.assertGreater(np.count_nonzero(resized_mask == 1), 0)
        self.assertGreater(np.count_nonzero(resized_mask == 2), 0)

    def test_split_records_preserves_strata_and_has_no_overlap(self) -> None:
        records: list[ml.ImageRecord] = []
        for index in range(5):
            records.append(make_record(index, (make_annotation(1, [1, 1, 2, 1, 2, 2, 1, 2]),)))
        for index in range(5, 10):
            records.append(make_record(index, (make_annotation(2, [1, 1, 2, 1, 2, 2, 1, 2]),)))
        for index in range(10, 15):
            records.append(make_record(index, ()))

        train, valid, test = ml.split_records(
            records,
            CATEGORY_TO_TRAIN_ID,
            val_size=0.2,
            test_size=0.2,
            seed=7,
            max_samples=None,
        )

        all_ids = [record.image_id for record in train + valid + test]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(len(train), 9)
        self.assertEqual(len(valid), 3)
        self.assertEqual(len(test), 3)
        for split in (train, valid, test):
            labels = {ml.image_stratification_label(record, CATEGORY_TO_TRAIN_ID) for record in split}
            self.assertEqual(labels, {"anode", "member", "background"})

    def test_absent_foreground_dice_losses_are_zero(self) -> None:
        logits = torch.zeros((1, 3, 3, 3), dtype=torch.float32, requires_grad=True)
        targets = torch.zeros((1, 3, 3), dtype=torch.long)

        self.assertEqual(float(ml.soft_dice_loss(logits, targets).detach()), 0.0)
        self.assertEqual(float(ml.binary_dice_loss(logits[:, 1], targets == 1).detach()), 0.0)

    def test_metrics_from_confusion_reports_anode_and_member_only(self) -> None:
        confusion = torch.tensor(
            [
                [5, 1, 0],
                [0, 3, 1],
                [0, 2, 6],
            ],
            dtype=torch.int64,
        )

        metrics = ml.metrics_from_confusion(confusion)

        self.assertEqual(set(metrics), {"anode", "member"})
        self.assertAlmostEqual(metrics["anode"]["iou"], 3 / 7)
        self.assertAlmostEqual(metrics["anode"]["f1"], 6 / 10)
        self.assertAlmostEqual(metrics["member"]["iou"], 6 / 9)
        self.assertAlmostEqual(metrics["member"]["f1"], 12 / 15)

    def test_train_augmentation_preserves_mask_labels_and_anode_pixels(self) -> None:
        random.seed(3)
        np.random.seed(3)
        image = np.full((12, 12, 3), 128, dtype=np.uint8)
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[2:10, 2:10] = 2
        mask[5:7, 5:7] = 1
        config = ml.AugmentationConfig(
            enabled=True,
            hflip_prob=1.0,
            affine_prob=1.0,
            max_rotate_deg=0.0,
            scale_jitter=0.0,
            translate_jitter=0.0,
            intensity_prob=1.0,
            noise_prob=0.0,
            blur_prob=0.0,
        )

        augmented_image, augmented_mask = ml.apply_train_augmentation(image, mask, config)

        self.assertEqual(augmented_image.shape, image.shape)
        self.assertEqual(augmented_mask.shape, mask.shape)
        self.assertEqual(set(np.unique(augmented_mask)).issubset({0, 1, 2}), True)
        self.assertGreater(np.count_nonzero(augmented_mask == 1), 0)

    def test_geometry_rejection_helpers_protect_tiny_targets(self) -> None:
        before = np.zeros((4, 4), dtype=np.uint8)
        before[1, 1] = 1
        after = np.zeros_like(before)

        self.assertTrue(ml.anode_would_be_erased(before, after))
        self.assertTrue(ml.foreground_would_drop_too_much(before, after))
        self.assertFalse(ml.foreground_would_drop_too_much(after, after))


if __name__ == "__main__":
    unittest.main()

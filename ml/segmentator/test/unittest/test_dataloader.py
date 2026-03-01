"""Test for DataLoader for segmetator"""

import unittest
from ml.segmentator.test.test_config import TestConfig
from ml.segmentator.dataloader import (VesselPatchSampler,
                                       FundusPatchDataset,
                                       FundusInferenceDataset)
from torch.utils.data import Dataset, DataLoader
import unittest
import torch
import os

config = TestConfig()


class _BaseFundusPatchDatasetTest:
    """
    Базовый набор тестов для Dataset

    содержит всю логику проверки
    не знает, как именно создан dataset
    """
    @classmethod
    def setUpClass(cls):
        cls.loader = DataLoader(
            cls.dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True
        )

    def test_batch_shapes(self):
        batch = next(iter(self.loader))
        self.assertEqual(
            batch['image'].shape,
            (config.BATCH_SIZE, 3, config.PATCH_SIZE, config.PATCH_SIZE)
        )
        self.assertEqual(
            batch["mask"].shape,
            (config.BATCH_SIZE, 1, config.PATCH_SIZE, config.PATCH_SIZE)
        )
        self.assertEqual(
            batch["skeleton"].shape,
            (config.BATCH_SIZE, 1, config.PATCH_SIZE, config.PATCH_SIZE)
        )

    def test_mask_is_binary(self):
        batch = next(iter(self.loader))
        unique = set(batch["mask"].unique().tolist())
        self.assertTrue(unique.issubset({0.0, 1.0}))

    def test_skeleton_is_binary(self):
        batch = next(iter(self.loader))
        unique = set(batch["skeleton"].unique().tolist())
        self.assertTrue(unique.issubset({0.0, 1.0}))

    def test_skeleton_is_sparse(self):
        batch = next(iter(self.loader))
        self.assertLess(batch["skeleton"].sum(), batch["mask"].sum())

    def test_no_empty_vessel_patches(self):
        for i, batch in enumerate(self.loader):
            self.assertGreater(
                batch["mask"].sum().item(),
                0,
                msg=f'Empty vassel patch found in batch {i}'
            )


class TestFundusPatchDatasetFull(
        _BaseFundusPatchDatasetTest, unittest.TestCase):
    """
    Проверяет:
        старый код не сломался
        image_ids=None работает как раньше
    """
    @classmethod
    def setUpClass(cls):
        """Запускается один раз перед всеми тестами"""
        cls.dataset = FundusPatchDataset(
            image_dir=config.TEST_IMAGE_DIR,
            mask_dir=config.TEST_MASK_DIR,
            image_ids=None,
            patch_size=config.PATCH_SIZE,
            min_vessel_ratio=config.MIN_VESSEL_RATIO,
            augment=False,
            debug=False
        )
        super().setUpClass()


class TestFundusPatchDatasetSubset(
        _BaseFundusPatchDatasetTest, unittest.TestCase):
    """
    Проверяет: 

    dataset действительно ограничен image_ids
    никакого leakage между фолдами
    """
    @classmethod
    def setUpClass(cls):
        all_ids = sorted(os.listdir(config.TEST_IMAGE_DIR))
        subset_ids = all_ids[: max(2, len(all_ids) // 3)]

        cls.dataset = FundusPatchDataset(
            image_dir=config.TEST_IMAGE_DIR,
            mask_dir=config.TEST_MASK_DIR,
            image_ids=subset_ids,
            patch_size=config.PATCH_SIZE,
            min_vessel_ratio=config.MIN_VESSEL_RATIO,
            augment=False,
            debug=False
        )
        cls.subset_ids = {os.path.splitext(f)[0] for f in subset_ids}
        super().setUpClass()

    def test_dataset_uses_inly_subset_ids(self):
        for i in range(len(self.dataset)):
            sample = self.dataset[i]
            self.assertIn(sample["image_id"], self.subset_ids)


class TestFundusInferenceDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Инициализируем даталоадер один раз для всех тестов
        """
        cls.dataset = FundusInferenceDataset(
            image_dir=config.TEST_IMAGE_DIR,
            patch_size=config.PATCH_SIZE,
            stride=config.STRIDE  # overlap специально
        )

        cls.loader = DataLoader(
            cls.dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS
        )

        cls.batch = next(iter(cls.loader))

    # Проверка ключей
    def test_batch_keys(self):
        expected_keys = {
            "image",
            "image_id",
            "coords",
            "patch_shape",
            "full_size"
        }
        self.assertEqual(set(self.batch.keys()), expected_keys)

    # 2. Проверка типов
    def test_batch_types(self):
        self.assertIsInstance(self.batch["image"], torch.Tensor)
        self.assertIsInstance(self.batch["coords"], torch.Tensor)
        self.assertIsInstance(self.batch["patch_shape"], torch.Tensor)
        self.assertIsInstance(self.batch["full_size"], torch.Tensor)
        self.assertIsInstance(self.batch["image_id"], list)

    # Проверка размеров батча
    def test_batch_shapes(self):
        B = self.batch["image"].shape[0]

        self.assertEqual(self.batch["image"].shape, (B, 3, 512, 512))
        self.assertEqual(self.batch["coords"].shape, (B, 2))
        self.assertEqual(self.batch["patch_shape"].shape, (B, 2))
        self.assertEqual(self.batch["full_size"].shape, (B, 2))

    # Проверка padding и реального размера
    def test_padding_and_patch_shape_consistency(self):
        for i in range(len(self.batch["image_id"])):
            h, w = self.batch["patch_shape"][i].tolist()
            y, x = self.batch["coords"][i].tolist()
            H, W = self.batch["full_size"][i].tolist()

            patch = self.batch["image"][i]

            # padded patch ВСЕГДА 512x512
            self.assertEqual(patch.shape, (3, 512, 512))

            # реальный размер не превышает patch_size
            self.assertTrue(h <= 512)
            self.assertTrue(w <= 512)

            # координаты валидны
            self.assertTrue(0 <= y < H)
            self.assertTrue(0 <= x < W)

    # Проверка, что overlap действительно есть
    def test_overlap_present(self):
        coords = self.batch["coords"][:, 1]  # x координаты
        diffs = coords[1:] - coords[:-1]

        # stride = 256, значит должны быть повторения областей
        self.assertTrue(torch.any(diffs < 512))


if __name__ == "__main__":
    unittest.main()

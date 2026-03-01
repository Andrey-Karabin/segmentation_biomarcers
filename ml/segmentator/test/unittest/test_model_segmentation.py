import unittest
import torch
import torch.nn.functional as F

from ml.segmentator.model_segmentation import SegFormerSegmentation, SegmentationLoss
from ml.segmentator.test.test_config import TestConfig
from ml.segmentator.dataloader import FundusPatchDataset
from torch.utils.data import DataLoader
from ml.segmentator.test.test_config import TestConfig


class TestSegFormerSegmentation(unittest.TestCase):
    """
    Tests for SegFormerSegmentation (image + skeleton -> vessel mask)
    """

    @classmethod
    def setUpClass(cls):
        """
        Initialization is performed once for the entire test class.
        """
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.config = TestConfig()

        cls.dataset = FundusPatchDataset(
            image_dir=cls.config.TEST_IMAGE_DIR,
            mask_dir=cls.config.TEST_MASK_DIR,
            patch_size=cls.config.PATCH_SIZE,
            augment=False
        )

        cls.loader = DataLoader(
            cls.dataset,
            batch_size=cls.config.BATCH_SIZE_SEG_MODEL,
            shuffle=True
        )

        cls.model = SegFormerSegmentation().to(cls.device)
        cls.criterion = SegmentationLoss()

    def test_forward_shape(self):
        """
        We check that forward:
        - accepts image + skeleton
        - returns a tensor of the correct shape [B, 1, H, W]
        """

        x = torch.randn(2, 3, 512, 512).to(self.device)
        skel = torch.randn(2, 1, 512, 512).to(self.device)

        y = self.model(x, skel)

        # Checking the basic contract
        self.assertEqual(y.shape, (2, 1, 128, 128),
                         "Model output must have shape [B, 1, 128, 128]")

    def test_training_step(self):
        """
        We're testing one full training step:
        - the data is correct
        - the loss is calculated
        - the loss is finite and positive
        - the gradients are flowing
        """

        batch = next(iter(self.loader))

        image = batch["image"].to(self.device)
        mask_gt = batch["mask"].to(self.device)
        skel_gt = batch["skeleton"].to(self.device)

        pred = self.model(image, skel_gt)

        # We'll sample it just like in a real pipeline.
        pred_up = F.interpolate(
            pred,
            size=mask_gt.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        loss, logs = self.criterion(pred_up, mask_gt, skel_gt)

        # loss must be finite
        self.assertTrue(torch.isfinite(loss))

        # loss must be positive
        self.assertGreater(loss.item(), 0)

        loss.backward()

        grad_norm = sum(
            p.grad.norm().item()
            for p in self.model.parameters()
            if p.grad is not None
        )

        # gradients flow
        self.assertGreater(grad_norm, 0)

    def test_upsampling_to_gt_size(self):
        """
        We check that the prediction is correctly upsampled to the GT size.
        """
        batch = next(iter(self.loader))

        image = batch["image"].to(self.device)
        mask_gt = batch["mask"].to(self.device)
        skel_gt = batch["skeleton"].to(self.device)

        out = self.model(image, skel_gt)

        out_up = F.interpolate(
            out,
            size=mask_gt.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        self.assertEqual(
            out_up.shape,
            mask_gt.shape,
            "Upsampled prediction must match GT mask shape"
        )


if __name__ == "__main__":
    unittest.main()

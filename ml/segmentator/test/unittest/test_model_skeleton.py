import unittest
import torch

from ml.segmentator.model_skeleton import SegFormerSkeleton
from ml.segmentator.model_skeleton import SkeletonLoss
from ml.segmentator.test.test_config import TestConfig
from ml.segmentator.dataloader import FundusPatchDataset
from torch.utils.data import DataLoader

config = TestConfig()


class TestSegFormerSkeleton(unittest.TestCase):
    """
    Unit tests for the SegFormerSkeleton model.

    Goal:
    - Verify the correctness of the forward-pass
    - Verify that the model is trainable (loss is calculated and backprop is working)
    """
    @classmethod
    def setUpClass(cls):
        """
        Prepare shared objects once for all tests
        """
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"

        cls.dataset = FundusPatchDataset(
            image_dir=config.TEST_IMAGE_DIR,
            mask_dir=config.TEST_MASK_DIR,
            patch_size=config.PATCH_SIZE,
            augment=False
        )

        cls.loader = DataLoader(
            cls.dataset,
            batch_size=config.BATCH_SIZE_SK_MODEL,
            shuffle=True
        )

        cls.model = SegFormerSkeleton(
            backbone=config.SEGFORMER_SKELETON
        ).to(cls.device)

        cls.criterion = SkeletonLoss(alpha=1.0, beta=1.0)

    def test_forward_shape(self):
        """
        We check that the forward-pass model:
        - accepts input [B,3,512,512]
        - returns skeleton [B,1,512,512]
        """
        x = torch.randn(2, 3, 512, 512).to(self.device)
        y = self.model(x)
        # We check that the output has exactly 1 channel (skeleton),
        # and the spatial size has not changed
        self.assertEqual(y.shape, (2, 1, 512, 512),
                         msg="Model output shape must be [B,1,512,512]")

    def test_training_step(self):
        """
        We check that:
        - the loss is calculated correctly
        - the loss is finite and positive
        - the gradients actually flow (the model is trainable)
        """
        batch = next(iter(self.loader))

        image = batch["image"].to(self.device)
        skeleton_gt = batch["skeleton"].to(self.device)

        pred = self.model(image)
        loss = self.criterion(pred, skeleton_gt)
        # Check 1:
        # loss is neither NaN nor Inf
        # Catches:
        # - numerical explosions
        # - division by zero
        # - incorrect normalization
        self.assertTrue(torch.isfinite(
            loss),  msg="Loss is NaN or Inf — numerical instability detected")
        # Check 2:
        # loss > 0
        # Catches:
        # - empty GT
        # - trivial zero loss function
        self.assertGreater(
            loss.item(), 0, msg="Loss should be positive for non-empty skeleton GT")

        loss.backward()
        # We calculate the norm of all gradients
        grad_norm = sum(
            p.grad.norm().item()
            for p in self.model.parameters()
            if p.grad is not None
        )
        # Check 3:
        # gradients actually pass through the model
        # Catches:
        # - detached graph
        # - requires_grad=False
        # - errors in forward
        self.assertGreater(
            grad_norm, 0, msg="No gradients flowing — model is not trainable")


if __name__ == "__main__":
    unittest.main()

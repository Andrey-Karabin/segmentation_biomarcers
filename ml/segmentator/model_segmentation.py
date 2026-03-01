import torch.nn.functional as F
import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation
from ml.segmentator.config import Config
from ml.segmentator.model_skeleton import clDiceLoss

config = Config()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SegFormerSegmentation(nn.Module):
    """
    SegFormer for vessel segmentation with skeleton-aware input
    Input  : RGB image + predicted skeleton (4 channels)
    Output : vessel probability map
    """

    def __init__(self, backbone="nvidia/segformer-b0-finetuned-ade-512-512"):
        super().__init__()

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            backbone,
            num_labels=1,
            ignore_mismatched_sizes=True
        )

        # Modify input projection: from 3 -> 4 channels
        old_conv = self.model.segformer.encoder.patch_embeddings[0].proj
        self.model.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding
        )

        # Initialize new channel reasonably
        with torch.no_grad():
            self.model.segformer.encoder.patch_embeddings[0].proj.weight[:,
                                                                         :3] = old_conv.weight
            self.model.segformer.encoder.patch_embeddings[0].proj.weight[:, 3:] = old_conv.weight.mean(
                dim=1, keepdim=True)

    def forward(self, image, skeleton):
        """
        image    : [B, 3, H, W]
        skeleton : [B, 1, H, W]
        """
        x = torch.cat([image, skeleton], dim=1)
        out = self.model(pixel_values=x).logits
        return out


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        return 1 - (2. * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )


class SegmentationLoss(nn.Module):
    def __init__(self, lambda_dice=1.0, lambda_bce=1.0, lambda_cldice=0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.cldice = clDiceLoss()

        self.l1 = lambda_dice
        self.l2 = lambda_bce
        self.l3 = lambda_cldice

    def forward(self, pred, mask_gt, skel_gt):
        loss_dice = self.dice(pred, mask_gt)
        loss_bce = self.bce(pred, mask_gt)
        loss_cldice = self.cldice(pred, skel_gt)

        total = (
            self.l1 * loss_dice +
            self.l2 * loss_bce +
            self.l3 * loss_cldice
        )

        return total, {
            "dice": loss_dice.item(),
            "bce": loss_bce.item(),
            "cldice": loss_cldice.item()
        }

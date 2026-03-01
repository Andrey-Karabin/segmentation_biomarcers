import torch.nn.functional as F
import torch
import torch.nn as nn
from transformers import SegformerModel
from ml.segmentator.config import Config

config = Config()


class SegFormerSkeleton(nn.Module):
    """
    SegFormer-based skeleton prediction network
    """

    def __init__(
        self,
        backbone=config.SEGFORMER_SKELETON,
        in_channels=3,
        pretrained=True
    ):
        super().__init__()

        # Encoder (MiT)
        self.encoder = SegformerModel.from_pretrained(
            backbone,
            ignore_mismatched_sizes=True
        ) if pretrained else SegformerModel.from_config(backbone)

        hidden_sizes = self.encoder.config.hidden_sizes

        # Decoder head (simple but effective)
        self.decoder = nn.Sequential(
            nn.Conv2d(sum(hidden_sizes), 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        """
        x: [B, 3, H, W]
        return: skeleton logits [B, 1, H, W]
        """

        B, C, H, W = x.shape

        encoder_outputs = self.encoder(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True
        )

        features = encoder_outputs.hidden_states  # list of 4 tensors

        # Все фичи приводим к H/4
        target_size = features[0].shape[-2:]

        feats = [
            nn.functional.interpolate(
                f,
                size=target_size,
                mode="bilinear",
                align_corners=False
            )
            for f in features
        ]

        fused = torch.cat(feats, dim=1)
        skel_logits_lowres = self.decoder(fused)

        skel_logits = nn.functional.interpolate(
            skel_logits_lowres,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

        return skel_logits


class SkeletonBCELoss(nn.Module):
    def __init__(self, pos_weight=5.0):
        """
        pos_weight > 1 increases importance of skeleton pixels
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight)
        )

    def forward(self, pred, target):
        """
        pred   : [B, 1, H, W] logits
        target: [B, 1, H, W] binary skeleton
        """
        return self.bce(pred, target)


def soft_erode(img):
    p1 = -F.max_pool2d(-img, (3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(img):
    return F.max_pool2d(img, 3, stride=1, padding=1)


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iters=10):
    """
    Differentiable skeleton approximation
    """
    skel = torch.zeros_like(img)
    for _ in range(iters):
        opened = soft_open(img)
        delta = F.relu(img - opened)
        skel += delta
        img = soft_erode(img)
    return skel


class clDiceLoss(nn.Module):
    def __init__(self, iters=10, smooth=1e-6):
        super().__init__()
        self.iters = iters
        self.smooth = smooth

    def forward(self, pred, target):
        """
        pred   : [B, 1, H, W] sigmoid probabilities
        target : [B, 1, H, W] binary mask
        """
        pred = torch.sigmoid(pred)

        skel_pred = soft_skel(pred, self.iters)
        skel_gt = soft_skel(target.float(), self.iters)

        tprec = (pred * skel_gt).sum() / (skel_gt.sum() + self.smooth)
        tsens = (target * skel_pred).sum() / (skel_pred.sum() + self.smooth)

        cldice = (2 * tprec * tsens) / (tprec + tsens + self.smooth)
        return 1 - cldice


class SkeletonLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.bce = SkeletonBCELoss(pos_weight=5.0)
        self.cldice = clDiceLoss()

        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        loss_bce = self.bce(pred, target)
        loss_cldice = self.cldice(pred, target)

        return self.alpha * loss_bce + self.beta * loss_cldice

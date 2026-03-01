import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ml.segmentator.dataloader import FundusPatchDataset
from ml.segmentator.model_skeleton import SegFormerSkeleton
from ml.segmentator.model_segmentation import SegFormerSegmentation, SegmentationLoss
from ml.segmentator.config import Config
from ml.segmentator.splits import stratifield_kfold_split

# CONFIG & DEVICE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
config = Config()

# UTILS


def upsample_to_gt(pred, gt):
    """Upsample prediction to GT spatial size"""
    return F.interpolate(pred, size=gt.shape[2:], mode="bilinear", align_corners=False)


@torch.no_grad()
def cldice_metric(pred_logits, skel_gt):
    """Compute clDice metric (0..1)"""
    pred_bin = (torch.sigmoid(pred_logits) > 0.5).float()
    tp = (pred_bin * skel_gt).sum()
    fp = (pred_bin * (1 - skel_gt)).sum()
    fn = ((1 - pred_bin) * skel_gt).sum()
    eps = 1e-6
    return ((2 * tp + eps) / (2 * tp + fp + fn + eps)).item()


# TEST ONE BATCH
splits = list(stratifield_kfold_split(
    config.TRAIN_IMAGE_DIR,
    config.TRAIN_MASK_DIR,
    n_splits=config.N_FOLDS
))

fold, train_ids, val_ids = splits[0]

# Dataset + DataLoader
val_ds = FundusPatchDataset(
    config.TRAIN_IMAGE_DIR,
    config.TRAIN_MASK_DIR,
    image_ids=val_ids,
    patch_size=config.PATCH_SIZE,
    augment=False
)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

# Models
print("Initializing skeleton model...")
model_skel = SegFormerSkeleton(backbone=config.SEGFORMER_SKELETON).to(DEVICE)
skel_ckpt = torch.load(os.path.join(
    config.SAVE_DIR, f"fold_{fold}", "skeleton_best.pth"), map_location=DEVICE)
model_skel.load_state_dict(skel_ckpt["model_state_dict"])
model_skel.eval()
for p in model_skel.parameters():
    p.requires_grad = False

print("Initializing segmentation model...")
model_seg = SegFormerSegmentation().to(DEVICE)
criterion = SegmentationLoss(
    lambda_dice=1.0, lambda_bce=1.0, lambda_cldice=0.5)

# Take one batch
batch = next(iter(val_loader))
img = batch["image"].to(DEVICE)
mask_gt = batch["mask"].to(DEVICE)

print("Input image shape:", img.shape)
print("GT mask shape:", mask_gt.shape)

# Skeleton prediction
with torch.no_grad():
    skel_pred = model_skel(img)
    skel_pred = upsample_to_gt(skel_pred, mask_gt)

print("GT skeleton shape:", mask_gt.shape)
print("Skeleton pred shape:", skel_pred.shape,
      f"min/max: {skel_pred.min().item()} / {skel_pred.max().item()}")

# Segmentation prediction
with torch.no_grad():
    pred = model_seg(img, skel_pred)
    pred = upsample_to_gt(pred, mask_gt)

print("Segmentation pred shape:", pred.shape,
      f"min/max: {pred.min().item()} / {pred.max().item()}")

# Compute loss
loss, logs = criterion(pred, mask_gt, skel_pred)
print("Loss value:", loss.item())
print("Logs (raw):", logs)

# Compute clDice metric correctly
cldice_val = cldice_metric(pred, skel_pred)
print("clDice metric (0..1):", cldice_val)

# Check tensor types
print("Types:", type(img), type(mask_gt), type(pred), type(skel_pred))

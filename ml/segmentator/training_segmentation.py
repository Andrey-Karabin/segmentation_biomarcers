import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

from ml.segmentator.dataloader import FundusPatchDataset
from ml.segmentator.model_skeleton import SegFormerSkeleton
from ml.segmentator.model_segmentation import SegFormerSegmentation, SegmentationLoss
from ml.segmentator.config import Config
from ml.segmentator.splits import stratifield_kfold_split

from torch.cuda.amp import autocast, GradScaler

# CONFIG & DEVICE

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
config = Config()

SAVE_ROOT = config.SAVE_DIR_SEG
os.makedirs(SAVE_ROOT, exist_ok=True)

# Путь к JSON с найденными параметрами
SEARCH_JSON = os.path.join(
    config.PATH_SEARCH, "segmentation_search_results.json")
with open(SEARCH_JSON, "r") as f:
    search_results = json.load(f)

# Выбираем лучшую комбинацию (по best_cldice)
best_cfg = max(search_results, key=lambda x: x["best_cldice"])
LR = best_cfg["lr"]
LAMBDA_CLDICE = 0.25

print(f"Using best params: LR={LR}, lambda_cldice={LAMBDA_CLDICE}")
# UTILS


def upsample_to_gt(pred, gt):
    """Upsample prediction to GT spatial size"""
    return F.interpolate(
        pred,
        size=gt.shape[2:],
        mode="bilinear",
        align_corners=False
    )


# K-FOLD TRAINING

for fold, train_ids, val_ids in stratifield_kfold_split(
        image_dir=config.TRAIN_IMAGE_DIR,
        mask_dir=config.TRAIN_MASK_DIR,
        n_splits=config.N_FOLDS):

    print("\n" + "=" * 60)
    print(f"FOLD {fold}")
    print("=" * 60)

    fold_dir = os.path.join(SAVE_ROOT, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    # DATASETS

    train_dataset = FundusPatchDataset(
        image_dir=config.TRAIN_IMAGE_DIR,
        mask_dir=config.TRAIN_MASK_DIR,
        image_ids=train_ids,
        patch_size=config.PATCH_SIZE,
        augment=True
    )

    val_dataset = FundusPatchDataset(
        image_dir=config.TRAIN_IMAGE_DIR,
        mask_dir=config.TRAIN_MASK_DIR,
        image_ids=val_ids,
        patch_size=config.PATCH_SIZE,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(DEVICE == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(DEVICE == "cuda")
    )

    # MODELS

    #  Skeleton model (frozen)
    model_skel = SegFormerSkeleton(
        backbone=config.SEGFORMER_SKELETON
    ).to(DEVICE)

    skel_ckpt_path = os.path.join(
        config.SAVE_DIR, f"fold_{fold}", "skeleton_best.pth"
    )

    skel_ckpt = torch.load(skel_ckpt_path, map_location=DEVICE)
    model_skel.load_state_dict(skel_ckpt["model_state_dict"])
    model_skel.eval()

    for p in model_skel.parameters():
        p.requires_grad = False

    #  Segmentation model
    model_seg = SegFormerSegmentation().to(DEVICE)

    # LOSS / OPTIM / AMP

    criterion = SegmentationLoss(
        lambda_dice=1.0,
        lambda_bce=1.0,
        lambda_cldice=LAMBDA_CLDICE
    )

    optimizer = torch.optim.AdamW(
        model_seg.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    # TRAINING LOOP

    best_val_metric = -1.0

    for epoch in range(1, config.EPOCHS_SEG + 1):
        print(f"\n[Fold {fold}] Epoch {epoch}/{config.EPOCHS_SEG}")

        # TRAIN

        model_seg.train()
        train_logs = defaultdict(float)

        for batch in tqdm(train_loader, desc="Train"):
            img = batch["image"].to(DEVICE)
            mask_gt = batch["mask"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                skel_pred = model_skel(img)
                skel_pred = upsample_to_gt(skel_pred, mask_gt)

            with autocast(enabled=(DEVICE == "cuda")):
                pred = model_seg(img, skel_pred)
                pred = upsample_to_gt(pred, mask_gt)

                loss, logs = criterion(pred, mask_gt, skel_pred)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_logs["loss"] += loss.item()
            for k, v in logs.items():
                train_logs[k] += v

        for k in train_logs:
            train_logs[k] /= len(train_loader)

        # VALIDATION

        model_seg.eval()
        val_logs = defaultdict(float)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val"):
                img = batch["image"].to(DEVICE)
                mask_gt = batch["mask"].to(DEVICE)

                skel_pred = model_skel(img)
                skel_pred = upsample_to_gt(skel_pred, mask_gt)

                pred = model_seg(img, skel_pred)
                pred = upsample_to_gt(pred, mask_gt)

                loss, logs = criterion(pred, mask_gt, skel_pred)

                val_logs["loss"] += loss.item()
                for k, v in logs.items():
                    val_logs[k] += v

        for k in val_logs:
            val_logs[k] /= len(val_loader)

        print(f"[Fold {fold}] TRAIN:", dict(train_logs))
        print(f"[Fold {fold}] VAL  :", dict(val_logs))

        # SAVE BEST MODEL

        metric_to_track = val_logs.get("cldice", val_logs["dice"])

        if metric_to_track > best_val_metric:
            best_val_metric = metric_to_track

            save_path = os.path.join(fold_dir, "segmentation_best.pth")
            torch.save(
                {
                    "model_state_dict": model_seg.state_dict(),
                    "epoch": epoch,
                    "val_metrics": dict(val_logs),
                    "config": vars(config)
                },
                save_path
            )

            with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
                json.dump(
                    {
                        "best_val_metric": best_val_metric,
                        "epoch": epoch,
                        "train": dict(train_logs),
                        "val": dict(val_logs)
                    },
                    f,
                    indent=4
                )

            print(f" Saved BEST model to {save_path}")

    # optional: free GPU memory between folds
    del model_seg
    torch.cuda.empty_cache()

import os
import json
import itertools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.segmentator.config import Config
from ml.segmentator.dataloader import FundusPatchDataset
from ml.segmentator.model_skeleton import SegFormerSkeleton
from ml.segmentator.model_segmentation import SegFormerSegmentation, SegmentationLoss
from ml.segmentator.splits import stratifield_kfold_split

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

# CONFIG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
config = Config()

SEARCH_EPOCHS = config.SEARCH_EPOCH
BATCH_SIZE = config.BATCH_SIZE

# Search space
LR_LIST = config.LR_LIST_SEG
LAMBDA_CLDICE_LIST = config.CL_DICE_LIST

SAVE_DIR = config.PATH_SEARCH
os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_JSON = os.path.join(SAVE_DIR, "segmentation_search_results.json")


# UTILS

def upsample_to_gt(pred, gt):
    return F.interpolate(
        pred, size=gt.shape[2:], mode="bilinear", align_corners=False
    )


@torch.no_grad()
def segmentation_cldice_score(pred, mask_gt, skel_gt):
    """
    pred      : logits [B,1,H,W]
    mask_gt  : GT mask [B,1,H,W]
    skel_gt  : GT skeleton [B,1,H,W]
    """
    pred_bin = (torch.sigmoid(pred) > 0.5).float()

    tp = (pred_bin * skel_gt).sum()
    fp = (pred_bin * (1 - skel_gt)).sum()
    fn = ((1 - pred_bin) * skel_gt).sum()

    eps = 1e-6
    return (2 * tp + eps) / (2 * tp + fp + fn + eps)


# ONE EXPERIMENT

def run_one_experiment(
    lr,
    lambda_cldice,
    train_ids,
    val_ids,
    fold
):
    # DATASETS
    train_ds = FundusPatchDataset(
        config.TRAIN_IMAGE_DIR,
        config.TRAIN_MASK_DIR,
        image_ids=train_ids,
        patch_size=config.PATCH_SIZE,
        augment=True
    )

    val_ds = FundusPatchDataset(
        config.TRAIN_IMAGE_DIR,
        config.TRAIN_MASK_DIR,
        image_ids=val_ids,
        patch_size=config.PATCH_SIZE,
        augment=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False
    )

    # MODELS
    model_skel = SegFormerSkeleton(
        backbone=config.SEGFORMER_SKELETON
    ).to(DEVICE)

    skel_ckpt = torch.load(
        os.path.join(config.SAVE_DIR, f"fold_{fold}", "skeleton_best.pth"),
        map_location=DEVICE
    )
    model_skel.load_state_dict(skel_ckpt["model_state_dict"])
    model_skel.eval()
    for p in model_skel.parameters():
        p.requires_grad = False

    model_seg = SegFormerSegmentation().to(DEVICE)

    #  LOSS
    criterion = SegmentationLoss(
        lambda_dice=1.0,
        lambda_bce=1.0,
        lambda_cldice=lambda_cldice
    )

    optimizer = torch.optim.AdamW(
        model_seg.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    best_val_cldice = 0.0

    # TRAIN LOOP

    for epoch in range(SEARCH_EPOCHS):
        model_seg.train()

        for batch in train_loader:
            img = batch["image"].to(DEVICE)
            mask_gt = batch["mask"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                skel_pred = model_skel(img)
                skel_pred = upsample_to_gt(skel_pred, mask_gt)

            pred = model_seg(img, skel_pred)
            pred = upsample_to_gt(pred, mask_gt)

            loss, _ = criterion(pred, mask_gt, skel_pred)
            loss.backward()
            optimizer.step()

        #  VALIDATION
        model_seg.eval()
        scores = []

        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(DEVICE)
                mask_gt = batch["mask"].to(DEVICE)

                skel_pred = model_skel(img)
                skel_pred = upsample_to_gt(skel_pred, mask_gt)

                pred = model_seg(img, skel_pred)
                pred = upsample_to_gt(pred, mask_gt)

                score = segmentation_cldice_score(
                    pred, mask_gt, skel_pred
                )
                scores.append(score)

        mean_cldice = torch.stack(scores).mean().item()
        best_val_cldice = max(best_val_cldice, mean_cldice)

    return best_val_cldice


# MAIN SEARCH

def main():
    splits = list(stratifield_kfold_split(
        config.TRAIN_IMAGE_DIR,
        config.TRAIN_MASK_DIR,
        n_splits=config.N_FOLDS
    ))

    fold, train_ids, val_ids = splits[0]  # только 1 fold

    results = []

    for lr, lambda_cldice in itertools.product(LR_LIST, LAMBDA_CLDICE_LIST):
        print(f"Testing lr={lr}, lambda_cldice={lambda_cldice}")
        cldice_val = run_one_experiment(
            lr, lambda_cldice, train_ids, val_ids, fold)
        print(f" → Best clDice: {cldice_val:.4f}")

        results.append({
            "lr": lr,
            "lambda_cldice": lambda_cldice,
            "best_cldice": cldice_val
        })

    # Сохраняем всё в JSON один раз
    SAVE_JSON = config.RESULT_PATH_SEG
    os.makedirs(os.path.dirname(SAVE_JSON), exist_ok=True)

    with open(SAVE_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # Выводим лучший результат
    best_config = max(results, key=lambda x: x["best_cldice"])
    print("\nBest config:", best_config)


if __name__ == "__main__":
    main()

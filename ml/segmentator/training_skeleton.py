import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
from ml.segmentator.dataloader import FundusPatchDataset
from ml.segmentator.model_skeleton import SegFormerSkeleton, SkeletonLoss
from ml.segmentator.config import Config
from ml.segmentator.splits import stratifield_kfold_split
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
config = Config()

result_path = config.RESULTS_PATH
with open(result_path) as f:
    best_cfg = max(json.load(f), key=lambda x: x["f1"])

LR = best_cfg["lr"]
ALPHA = best_cfg["alpha"]
BETA = best_cfg["beta"]


def skeleton_precision(pred, gt, eps=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    tp = (pred * gt).sum()
    return tp / (pred.sum() + eps)


for fold, train_ids, val_ids in stratifield_kfold_split(image_dir=config.TRAIN_IMAGE_DIR, mask_dir=config.TRAIN_MASK_DIR, n_splits=config.N_FOLDS):
    print(f"Folds {fold}")
    print(f"{'='*60}")
    SAVE_DIR: str = 'ml/data/checkpoints/skeleton'
    fold_dir = os.path.join(config.SAVE_DIR, f'fold_{fold}')
    os.makedirs(fold_dir, exist_ok=True)

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
        augment=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    model = SegFormerSkeleton(
        backbone=config.SEGFORMER_SKELETON
    ).to(DEVICE)

    criterion = SkeletonLoss(alpha=ALPHA, beta=BETA)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    best_val_loss = float("inf")
    scaler = GradScaler(enabled=(DEVICE == "cuda"))
    
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n [Fold {fold}] Epoch {epoch}/{config.EPOCHS}")

        if epoch == 1:
            print(f"[Fold {fold}] Train size: {len(train_dataset)}")
            print(f"[Fold {fold}] Val size  : {len(val_dataset)}")

       # Train
        model.train()

        train_loss = 0.0
        train_prec = 0.0

        pbar = tqdm(train_loader, desc="Train")

        for batch in pbar:
            image = batch["image"].to(DEVICE, non_blocking=True)
            skel_gt = batch["skeleton"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(DEVICE == "cuda")):
                pred = model(image)
                loss = criterion(pred, skel_gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                prec = skeleton_precision(pred, skel_gt)

            train_loss += loss.item()
            train_prec += prec.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "prec": f"{prec.item():.4f}"
            })


        train_loss /= len(train_loader)
        train_prec /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_prec = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val"):
                image = batch["image"].to(DEVICE)
                skel_gt = batch["skeleton"].to(DEVICE)

                pred = model(image)
                loss = criterion(pred, skel_gt)

                prec = skeleton_precision(pred, skel_gt)

                val_loss += loss.item()
                val_prec += prec.item()
        val_loss /= len(val_loader)
        val_prec /= len(val_loader)

        print(f"[Fold {fold}]")
        print(f"\n Train loss {train_loss:.4f}, prec: {train_prec:.4f}")
        print(f"\n Val loss: {val_loss:.4f}, prec: {val_prec:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(fold_dir,
                                     "skeleton_best.pth")
            # torch.save(model.state_dict(), save_path)
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": vars(config)
            },
                save_path)
            print(f" Fold {fold} Saved best model to {save_path}")

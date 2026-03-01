import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import pandas as pd

from ml.biomarcers.config import Config
from ml.biomarcers.dataloader import ImageMaskDataset
from ml.biomarcers.utils_loss import TverskyLoss

config = Config()
torch.multiprocessing.set_start_method("spawn", force=True)

# Dice score для мультикласса


@torch.no_grad()
def dice_score_fast(preds, targets, ignore_index=config.IGNORE_INDEX):
    """
    Быстрый Dice для мультикласса без one-hot, только hard labels
    preds: logits (B, C, H, W)
    targets: (B, H, W)
    """
    preds_labels = preds.argmax(dim=1)  # (B,H,W)
    num_classes = preds.shape[1]

    dice_sum = 0.0
    count = 0
    eps = 1e-6

    for cls in range(1, num_classes):
        pred_mask = (preds_labels == cls)
        target_mask = (targets == cls)

        # игнорируем пиксели ignore_index
        valid_mask = (targets != ignore_index)
        pred_mask = pred_mask & valid_mask
        target_mask = target_mask & valid_mask

        TP = (pred_mask & target_mask).sum().item()
        FP = (pred_mask & (~target_mask)).sum().item()
        FN = ((~pred_mask) & target_mask).sum().item()

        if TP + FP + FN == 0:  # если класс отсутствует
            continue

        dice_cls = (2 * TP + eps) / (2 * TP + FP + FN + eps)
        dice_sum += dice_cls
        count += 1

    if count == 0:
        return 0.0
    return dice_sum / count


# Основная функция обучения

def train_fold(train_folds, val_fold, patience=5):
    # Загружаем CSV
    train_dfs = [pd.read_csv(
        f"D:\\aspirantura\\PROF\\npy_article_fold\\train_article_fold_{f}.csv") for f in train_folds]
    df_train = pd.concat(train_dfs).reset_index(drop=True)
    df_val = pd.read_csv(
        f"D:\\aspirantura\\PROF\\npy_article_fold\\train_article_fold_{val_fold}.csv")

    # Datasets
    train_dataset = ImageMaskDataset(df_train, augment_prob=0.5)
    val_dataset = ImageMaskDataset(df_val, augment_prob=0.0)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)

    # Модель
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=config.NUM_CLASSES,
        ignore_mismatched_sizes=True
    ).to(config.DEVICE)

    optimizer = torch.optim.AdamW([
        # Backbone: low LR
        {'params': model.segformer.encoder.parameters(), 'lr': 1e-5},
        {'params': model.decode_head.parameters(), 'lr': 5e-4},        # Head: high LR
    ])
    gradient_accumulation_steps = 2
    num_training_steps = (len(train_loader) //
                          gradient_accumulation_steps) * config.EPOCHS

    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    ce_loss = torch.nn.CrossEntropyLoss(
        ignore_index=config.IGNORE_INDEX).to(config.DEVICE)
    tversky_loss = TverskyLoss(
        ignore_index=config.IGNORE_INDEX).to(config.DEVICE)

    def combined_loss(logits, targets):
        return ce_loss(logits, targets) + 2.0 * tversky_loss(logits, targets)

    scaler = torch.cuda.amp.GradScaler()

    best_dice = 0.0
    epochs_no_improve = 0

    # Checkpoint path
    checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, f"fold_{val_fold}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        loader_iter = tqdm(
            train_loader,
            desc=f"Train Folds: {train_folds}, Val Fold: {val_fold} | Epoch {epoch+1}",
            unit="batch")

        optimizer.zero_grad()
        for i, (imgs, masks) in enumerate(loader_iter):
            imgs = imgs.to(config.DEVICE)
            masks = masks.to(config.DEVICE)
            if epoch == 0 and i == 0:
                print("Unique values in mask:", torch.unique(masks))

            with torch.cuda.amp.autocast():
                logits = model(pixel_values=imgs).logits
                logits = F.interpolate(
                    logits, masks.shape[-2:], mode="bilinear", align_corners=False)
                loss = combined_loss(logits, masks)
                loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            running_loss += loss.item() * gradient_accumulation_steps
            loader_iter.set_postfix({"avg_loss": running_loss / (i+1)})

        epoch_loss = running_loss / len(train_loader)
        print(
            f"Epoch {epoch+1} finished, Avg Loss: {epoch_loss:.4f}, Time: {time.time()-start_time:.1f}s")

        # Валидация

        model.eval()
        all_dice = 0.0
        count = 0
        val_iter = tqdm(
            val_loader, desc=f"Validation Fold {val_fold} Epoch {epoch+1}", unit="batch")
        for imgs, masks in val_iter:
            imgs = imgs.to(config.DEVICE)
            masks = masks.to(config.DEVICE)
            with torch.no_grad():
                logits = model(pixel_values=imgs).logits
                logits = F.interpolate(
                    logits, masks.shape[-2:], mode="bilinear", align_corners=False)
                d = dice_score_fast(logits, masks)
                all_dice += d
                count += 1
            val_iter.set_postfix({"dice": all_dice / count})
        avg_dice = all_dice / count
        print(f"Validation Dice: {avg_dice:.4f}")

        # Сохраняем чекпоинт каждой эпохи

        checkpoint_path = os.path.join(
            checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
            "val_dice": avg_dice
        }, checkpoint_path)

        # Обновляем лучшую модель

        if avg_dice > best_dice:
            best_dice = avg_dice
            epochs_no_improve = 0
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
                "val_dice": avg_dice
            }, best_model_path)
            print(f"Best model updated! Dice: {best_dice:.4f}")
        else:
            epochs_no_improve += 1

        # Early stopping

        if epochs_no_improve >= patience:
            print(
                f"Early stopping at epoch {epoch+1} (no improvement {patience} epochs).")
            break

    print(f"Training finished. Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Задаём фолд вручную
    FOLD = 3

    # Определяем train и val фолды
    if FOLD == 1:
        train_folds = [1, 3]
        val_fold = 2
    elif FOLD == 2:
        train_folds = [1, 2]
        val_fold = 3
    elif FOLD == 3:
        train_folds = [2, 3]
        val_fold = 1
    else:
        raise ValueError("FOLD должен быть 1, 2 или 3")

    print(f"TRAIN FOLDS: {train_folds}, VAL FOLD: {val_fold}")

    train_fold(train_folds, val_fold, patience=5)

import torch
from ml.segmentator.config import Config
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from skimage.morphology import skeletonize
import cv2
import numpy as np
from ml.segmentator.dataloader import FundusInferenceDataset
from torch.utils.data import DataLoader
import os
from ml.segmentator.model_skeleton import SegFormerSkeleton
from tqdm import tqdm
from ml.segmentator.utils import (dice_score,
                                  iou_score,
                                  precision_score,
                                  recall_score,
                                  accuracy_score,
                                  f1_score,
                                  cldice_score)
from collections import defaultdict
import json

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

"""
путь к full-size изображениям
путь к GT маскам (оригинальным или уже скелетам)
путь к чекпоинтам моделей
путь для сохранения предсказаний
путь для метрик (json / csv)

[1] PATHS
   ↓
[2] GT → SKELETON (optional)
   ↓
[3] TEST DATASET + LOADER
   ↓
[4] FOR EACH FOLD:
      ├─ load model
      ├─ patch inference
      ├─ reconstruct full mask
      ├─ metrics per image
      ├─ save PNG
      ├─ save per-image metrics
   ↓
[5] AGGREGATE + BEST MODEL
"""

#  STEP 1: paths & device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
config = Config()

IMG_DIR = config.TEST_IMAGE_DIR
GT_MASK_DIR = config.TEST_MASK_DIR
CHECKPOINT_ROOT = config.SAVE_DIR
PRED_SAVE_ROOT = config.PRED_SAVE_DIR
METRICS_SAVE_DIR = config.METRICS_SAVE_DIR

os.makedirs(PRED_SAVE_ROOT, exist_ok=True)
os.makedirs(METRICS_SAVE_DIR, exist_ok=True)


#  STEP 2: GT → skeleton (once)
GT_SKELETON_DIR = os.path.join(METRICS_SAVE_DIR, "gt_skeleton")
os.makedirs(GT_SKELETON_DIR, exist_ok=True)

for fname in os.listdir(GT_MASK_DIR):
    gt = cv2.imread(os.path.join(GT_MASK_DIR, fname), cv2.IMREAD_GRAYSCALE)
    gt = (gt > 0).astype(np.uint8)
    gt_skel = skeletonize(gt).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(GT_SKELETON_DIR, fname), gt_skel)


#  STEP 3: inference dataset & loader
test_dataset = FundusInferenceDataset(
    image_dir=IMG_DIR,
    patch_size=config.PATCH_SIZE,
    stride=config.STRIDE
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=(DEVICE == "cuda")
)


def load_model_for_fold(fold_idx, device):
    model = SegFormerSkeleton(
        backbone=config.SEGFORMER_SKELETON
    ).to(device)

    ckpt_path = os.path.join(
        CHECKPOINT_ROOT,
        f"fold_{fold_idx}",
        "skeleton_best.pth"
    )

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def init_buffers():
    return {}, {}


def run_inference_and_metrics(fold_idx, model, test_loader, device, gt_skeleton_dir):
    """
    Patch-wise inference и метрики, без сохранения PNG
    """
    model.eval()
    pred_buffers = {}
    count_buffers = {}
    image_metrics = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Inference fold {fold_idx}"):

            images = batch["image"].to(device)
            image_ids = batch["image_id"]
            coords = batch["coords"]
            patch_shapes = batch["patch_shape"]
            full_sizes = batch["full_size"]

            preds = torch.sigmoid(model(images))  # [B,1,H,W]

            for i in range(images.size(0)):
                img_id = image_ids[i]
                y, x = coords[i].tolist()
                h, w = patch_shapes[i].tolist()
                H, W = full_sizes[i].tolist()

                if img_id not in pred_buffers:
                    pred_buffers[img_id] = torch.zeros((H, W), device=device)
                    count_buffers[img_id] = torch.zeros((H, W), device=device)

                pred_buffers[img_id][y:y+h, x:x+w] += preds[i, 0, :h, :w]
                count_buffers[img_id][y:y+h, x:x+w] += 1

    #  Compute metrics per image
    aggregate_metrics = defaultdict(list)

    for img_id in pred_buffers:
        prob_map = pred_buffers[img_id] / count_buffers[img_id]
        bin_map_np = (prob_map > 0.5).cpu().numpy().astype(np.uint8)

        # Load GT skeleton
        gt_path = os.path.join(gt_skeleton_dir, img_id)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt = (gt > 0).astype(np.uint8)

        # Calculate all metrics
        metrics = {
            "dice": dice_score(bin_map_np, gt),
            "iou": iou_score(bin_map_np, gt),
            "precision": precision_score(bin_map_np, gt),
            "recall": recall_score(bin_map_np, gt),
            "accuracy": accuracy_score(bin_map_np, gt),
            "f1": f1_score(bin_map_np, gt),
            "cldice": cldice_score(bin_map_np, gt)
        }

        image_metrics[img_id] = metrics

        for k in metrics:
            aggregate_metrics[k].append(metrics[k])

    # Aggregate per fold
    aggregate_metrics = {k: float(np.mean(v))
                         for k, v in aggregate_metrics.items()}

    return image_metrics, aggregate_metrics


def find_best_fold(all_fold_metrics, metric="cldice"):
    """
    all_fold_metrics: dict[fold_idx] = {"aggregate": {...}, "per_image": {...}}
    Возвращает fold_idx с максимальным metric
    """
    best_val = -1
    best_fold = None
    for fold_idx, fold_data in all_fold_metrics.items():
        val = fold_data["aggregate"].get(metric, -1)
        if val > best_val:
            best_val = val
            best_fold = fold_idx
    return best_fold


def run_inference_and_save(fold_idx, model, test_loader, device, save_pred_dir):
    """
    Полностью повторяет reconstruction, но сохраняет PNG маски
    """
    os.makedirs(save_pred_dir, exist_ok=True)

    model.eval()
    pred_buffers = {}
    count_buffers = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Inference fold {fold_idx}"):

            images = batch["image"].to(device)
            image_ids = batch["image_id"]
            coords = batch["coords"]
            patch_shapes = batch["patch_shape"]
            full_sizes = batch["full_size"]

            preds = torch.sigmoid(model(images))

            for i in range(images.size(0)):
                img_id = image_ids[i]
                y, x = coords[i].tolist()
                h, w = patch_shapes[i].tolist()
                H, W = full_sizes[i].tolist()

                if img_id not in pred_buffers:
                    pred_buffers[img_id] = torch.zeros((H, W), device=device)
                    count_buffers[img_id] = torch.zeros((H, W), device=device)

                pred_buffers[img_id][y:y+h, x:x+w] += preds[i, 0, :h, :w]
                count_buffers[img_id][y:y+h, x:x+w] += 1

    #  save PNG
    for img_id in pred_buffers:
        prob_map = pred_buffers[img_id] / count_buffers[img_id]
        bin_map = (prob_map > 0.5).float().cpu().numpy()

        save_path = os.path.join(
            save_pred_dir, img_id.replace(".png", "_skeleton.png"))
        cv2.imwrite(save_path, (bin_map*255).astype("uint8"))


all_fold_metrics = {}

for fold in range(config.N_FOLDS):
    model = load_model_for_fold(fold, DEVICE)

    per_image_metrics, aggregate_metrics = run_inference_and_metrics(
        fold, model, test_loader, DEVICE, GT_SKELETON_DIR
    )

    all_fold_metrics[fold] = {
        "per_image": per_image_metrics,
        "aggregate": aggregate_metrics
    }

# Сохраняем все метрики
with open(os.path.join(METRICS_SAVE_DIR, "all_fold_metrics.json"), "w") as f:
    json.dump(all_fold_metrics, f, indent=4)

# Находим лучший fold
best_fold = find_best_fold(all_fold_metrics, metric="cldice")
print("Лучший fold по clDice:", best_fold)

# Повторный inference + сохранение PNG только для лучшего fold
best_model = load_model_for_fold(best_fold, DEVICE)
save_pred_dir = os.path.join(PRED_SAVE_ROOT, f"fold_{best_fold}")
run_inference_and_save(best_fold, best_model,
                       test_loader, DEVICE, save_pred_dir)

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import numpy as np
import json
from collections import defaultdict

from ml.segmentator.config import Config
from ml.segmentator.dataloader import FundusInferenceDataset
from ml.segmentator.model_segmentation import SegFormerSegmentation
from ml.segmentator.model_skeleton import SegFormerSkeleton
from ml.segmentator.utils import (dice_score, iou_score, precision_score, recall_score,
                                  accuracy_score, f1_score, cldice_score)


# DEVICE & CONFIG

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

config = Config()

IMG_DIR = config.TEST_IMAGE_DIR
GT_MASK_DIR = config.TEST_MASK_DIR
CHECKPOINT_ROOT_SEG = config.SAVE_DIR_SEG
CHECKPOINT_ROOT_SKL = config.SAVE_DIR
PRED_SAVE_ROOT = config.PRED_SAVE_DIR
METRICS_SAVE_DIR = config.METRICS_SAVE_DIR_SEG

os.makedirs(PRED_SAVE_ROOT, exist_ok=True)
os.makedirs(METRICS_SAVE_DIR, exist_ok=True)


# Step 1: prepare GT skeletons

GT_SKELETON_DIR = os.path.join(METRICS_SAVE_DIR, "gt_skeleton")
os.makedirs(GT_SKELETON_DIR, exist_ok=True)

for fname in os.listdir(GT_MASK_DIR):
    gt = cv2.imread(os.path.join(GT_MASK_DIR, fname), cv2.IMREAD_GRAYSCALE)
    gt_bin = (gt > 0).astype(np.uint8)
    gt_skel = (F.interpolate(torch.from_numpy(gt_bin).unsqueeze(0).unsqueeze(0).float(),
                             size=(gt.shape[0], gt.shape[1]),
                             mode="nearest").squeeze().numpy() > 0).astype(np.uint8)
    cv2.imwrite(os.path.join(GT_SKELETON_DIR, fname), gt_skel*255)


# Step 2: dataset + loader

test_dataset = FundusInferenceDataset(
    image_dir=IMG_DIR,
    patch_size=config.PATCH_SIZE,
    stride=config.STRIDE
)
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=(DEVICE == "cuda")
)


# Step 3: helper functions

def load_fold_models(fold_idx):
    """Load skeleton + segmentation model for a given fold"""
    skel_model = SegFormerSkeleton(
        backbone=config.SEGFORMER_SKELETON).to(DEVICE)
    skel_ckpt_path = os.path.join(
        CHECKPOINT_ROOT_SKL, f"fold_{fold_idx}", "skeleton_best.pth")
    skel_ckpt = torch.load(skel_ckpt_path, map_location=DEVICE)
    skel_model.load_state_dict(skel_ckpt["model_state_dict"])
    skel_model.eval()
    for p in skel_model.parameters():
        p.requires_grad = False

    seg_model = SegFormerSegmentation().to(DEVICE)
    seg_ckpt_path = os.path.join(
        CHECKPOINT_ROOT_SEG, f"fold_{fold_idx}", "segmentation_best.pth")
    seg_ckpt = torch.load(seg_ckpt_path, map_location=DEVICE)
    seg_model.load_state_dict(seg_ckpt["model_state_dict"])
    seg_model.eval()

    return seg_model, skel_model


def patch_inference(seg_model, skel_model, loader):
    """Patch-wise inference с буферами для full-image и debug-принтами"""
    pred_buffers = {}
    count_buffers = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Patch inference")):
            imgs = batch["image"].to(DEVICE)
            img_ids = batch["image_id"]
            coords_list = batch["coords"]
            patch_shapes_list = batch["patch_shape"]
            full_sizes_list = batch["full_size"]

            # Skeleton prediction
            skel_pred = skel_model(imgs)

            skel_pred_up = F.interpolate(
                skel_pred, size=(imgs.shape[2], imgs.shape[3]),
                mode="bilinear", align_corners=False
            )

            # Segmentation prediction
            seg_pred = seg_model(imgs, skel_pred_up)

            # Upsample segmentation до размера patch (patch_shape)
            seg_pred_up_list = []
            for i in range(imgs.size(0)):
                h_patch, w_patch = patch_shapes_list[i].tolist()
                seg_patch = F.interpolate(
                    seg_pred[i:i+1], size=(h_patch, w_patch),
                    mode="bilinear", align_corners=False
                )
                seg_pred_up_list.append(seg_patch)

            seg_pred_up = torch.cat(seg_pred_up_list, dim=0)
            seg_pred_sigmoid = torch.sigmoid(seg_pred_up)

            # вставка в буфер full-image
            for i in range(imgs.size(0)):
                img_id = img_ids[i]
                y, x = coords_list[i].tolist()
                h_patch, w_patch = patch_shapes_list[i].tolist()
                H, W = full_sizes_list[i].tolist()

                if img_id not in pred_buffers:
                    pred_buffers[img_id] = torch.zeros((H, W), device=DEVICE)
                    count_buffers[img_id] = torch.zeros((H, W), device=DEVICE)

                # Проверка размеров перед вставкой
                ph, pw = seg_pred_sigmoid[i, 0].shape
                if ph != h_patch or pw != w_patch:
                    print(f"  WARNING: patch size mismatch for {img_id}: "
                          f"seg_pred {ph,pw}, expected {h_patch,w_patch}")

                pred_buffers[img_id][y:y+h_patch, x:x +
                                     w_patch] += seg_pred_sigmoid[i, 0]
                count_buffers[img_id][y:y+h_patch, x:x+w_patch] += 1

    return pred_buffers, count_buffers


def compute_metrics(pred_buffers, count_buffers, gt_dir):
    """Compute all metrics per image"""
    per_image_metrics = {}
    aggregate_metrics = defaultdict(list)

    for img_id, pred_buffer in pred_buffers.items():
        prob_map = (pred_buffer / count_buffers[img_id]).cpu().numpy()
        bin_map = (prob_map > 0.5).astype(np.uint8)

        gt = cv2.imread(os.path.join(gt_dir, img_id), cv2.IMREAD_GRAYSCALE)
        gt = (gt > 0).astype(np.uint8)

        metrics = {
            "dice": dice_score(bin_map, gt),
            "iou": iou_score(bin_map, gt),
            "precision": precision_score(bin_map, gt),
            "recall": recall_score(bin_map, gt),
            "accuracy": accuracy_score(bin_map, gt),
            "f1": f1_score(bin_map, gt),
            "cldice": cldice_score(bin_map, gt)
        }

        per_image_metrics[img_id] = metrics
        for k, v in metrics.items():
            aggregate_metrics[k].append(v)

    aggregate_metrics = {k: float(np.mean(v))
                         for k, v in aggregate_metrics.items()}
    return per_image_metrics, aggregate_metrics


def save_predictions(pred_buffers, count_buffers, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for img_id, pred_buffer in pred_buffers.items():
        prob_map = (pred_buffer / count_buffers[img_id]).cpu().numpy()
        bin_map = (prob_map > 0.5).astype(np.uint8)
        save_path = os.path.join(save_dir, img_id.replace(".png", "_pred.png"))
        cv2.imwrite(save_path, (bin_map*255).astype(np.uint8))


# Step 4: run inference + collect metrics for all folds
all_fold_metrics = {}

for fold in range(config.N_FOLDS):
    print(f"\n=== Fold {fold} ===")
    seg_model, skel_model = load_fold_models(fold)
    pred_buffers, count_buffers = patch_inference(
        seg_model, skel_model, test_loader)
    per_img, agg = compute_metrics(
        pred_buffers, count_buffers, GT_SKELETON_DIR)
    all_fold_metrics[fold] = {"per_image": per_img, "aggregate": agg}

# Save metrics
with open(os.path.join(METRICS_SAVE_DIR, "all_fold_metrics.json"), "w") as f:
    json.dump(all_fold_metrics, f, indent=4)

# Найти лучший fold по clDice и сохранить предсказания
best_fold = max(all_fold_metrics.items(),
                key=lambda x: x[1]["aggregate"]["cldice"])[0]
print("Best fold by clDice:", best_fold)

best_seg_model, best_skel_model = load_fold_models(best_fold)
pred_buffers, count_buffers = patch_inference(
    best_seg_model, best_skel_model, test_loader)
save_dir = os.path.join(PRED_SAVE_ROOT, f"fold_{best_fold}")
save_predictions(pred_buffers, count_buffers, save_dir)

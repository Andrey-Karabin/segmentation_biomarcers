import os
import numpy as np
from skimage.io import imread
from skimage.morphology import skeletonize
from skimage.measure import label
from tqdm import tqdm
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from ml.segmentator.dataloader import FundusPatchDataset
from ml.segmentator.model_skeleton import SegFormerSkeleton, SkeletonLoss
from ml.segmentator.config import Config
from ml.segmentator.model_segmentation import SegFormerSegmentation, SegmentationLoss

# Папки

config = Config()

# Метрики


def remove_small_components(mask, min_size=100):
    labeled = label(mask, connectivity=2)
    out = np.zeros_like(mask)
    for cc in range(1, labeled.max() + 1):
        component = (labeled == cc)
        if component.sum() >= min_size:
            out[component] = 1
    return out


def dice_score(pred, gt, eps=1e-6):
    intersection = (pred * gt).sum()
    return (2 * intersection + eps) / (pred.sum() + gt.sum() + eps)


def cldice_score(pred, gt):
    skel_pred = skeletonize(pred)
    skel_gt = skeletonize(gt)

    tprec = (pred & skel_gt).sum() / (skel_gt.sum() + 1e-6)
    tsens = (gt & skel_pred).sum() / (skel_pred.sum() + 1e-6)

    return 2 * tprec * tsens / (tprec + tsens + 1e-6)


def betti_0(mask):
    return label(mask, connectivity=2).max()


def euler_characteristic(mask):
    labeled = label(mask, connectivity=2)
    num_components = labeled.max()
    inv = np.logical_not(mask)
    holes = label(inv, connectivity=2)
    hole_count = holes.max() - 1  # фон
    return num_components - hole_count


def betti_1(mask):
    return betti_0(mask) - euler_characteristic(mask)


def topology_errors(pred, gt):
    beta0_p = betti_0(pred)
    beta1_p = betti_1(pred)
    chi_p = beta0_p - beta1_p

    beta0_g = betti_0(gt)
    beta1_g = betti_1(gt)
    chi_g = beta0_g - beta1_g

    return {
        "beta0_error": abs(beta0_p - beta0_g),
        "beta1_error": abs(beta1_p - beta1_g),
        "euler_error": abs(chi_p - chi_g)
    }

# Подсчёт метрик


dice_list = []
cldice_list = []
beta0_errors = []
beta1_errors = []
euler_errors = []

pred_files = sorted(os.listdir(config.SAVE_DIR_PREDICTION_MASK))

for fname in tqdm(pred_files):
    # fname = "001_G_pred.png"
    img_id = os.path.splitext(fname)[0].replace(
        "_pred", "")  # убираем _pred, чтобы получить GT
    pred_path = os.path.join(config.SAVE_DIR_PREDICTION_MASK, fname)
    # теперь ищем правильный GT
    gt_path = os.path.join(config.TEST_MASK_DIR, f"{img_id}")

    pred = imread(pred_path)
    gt = imread(gt_path)

    # Если цветное изображение, конвертируем в бинарную маску
    if pred.ndim == 3:
        pred = pred[:, :, 0]
    if gt.ndim == 3:
        gt = gt[:, :, 0]

    pred_bin = (pred > 0.5).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    # Удаляем маленькие компоненты
    pred_bin = remove_small_components(pred_bin, min_size=100)

    # Метрики
    dice_list.append(dice_score(pred_bin, gt_bin))
    cldice_list.append(cldice_score(pred_bin, gt_bin))
    topo_errs = topology_errors(pred_bin, gt_bin)
    beta0_errors.append(topo_errs["beta0_error"])
    beta1_errors.append(topo_errs["beta1_error"])
    euler_errors.append(topo_errs["euler_error"])

# Средние метрики по датасету

print("=== Average Metrics ===")
print(f"Dice: {np.mean(dice_list):.4f}")
print(f"clDice: {np.mean(cldice_list):.4f}")
print(f"Beta0 error: {np.mean(beta0_errors):.2f}")
print(f"Beta1 error: {np.mean(beta1_errors):.2f}")
print(f"Euler characteristic error: {np.mean(euler_errors):.2f}")

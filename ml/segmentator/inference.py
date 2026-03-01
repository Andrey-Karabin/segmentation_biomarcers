import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from ml.segmentator.dataloader import FundusInferenceDataset
from ml.segmentator.model_skeleton import SegFormerSkeleton
from ml.segmentator.config import Config
from ml.segmentator.model_segmentation import SegFormerSegmentation
import numpy as np
import os
import matplotlib.pyplot as plt
from ml.segmentator.utils import load_models
config = Config()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(config.SAVE_DIR, exist_ok=True)


def reconstruct_image_from_patches(patches, coords_list, patch_shapes, full_size):
    """
    Собирает полное изображение из патчей с учётом padding и overlap.

    patches       : list of torch.Tensor или np.array, каждый [C,H,W] или [H,W]
    coords_list   : list of (y,x) координат верхнего левого угла патча
    patch_shapes  : list of (h,w) реальных размеров патча до padding
    full_size     : (H,W) размер исходного изображения

    return        : np.array, собранное изображение [H,W] или [H,W,C]
    """
    # Определяем размер канала
    first_patch = patches[0]
    if isinstance(first_patch, torch.Tensor):
        first_patch = first_patch.cpu().numpy()
    if first_patch.ndim == 3:
        C, _, _ = first_patch.shape
        full_image = np.zeros(
            (full_size[0], full_size[1], C), dtype=np.float32)
        count = np.zeros((full_size[0], full_size[1], C), dtype=np.float32)
    else:
        full_image = np.zeros(full_size, dtype=np.float32)
        count = np.zeros(full_size, dtype=np.float32)

    for i, patch in enumerate(patches):
        if isinstance(patch, torch.Tensor):
            patch = patch.cpu().numpy()

        y, x = coords_list[i]
        h, w = patch_shapes[i]

        # Обрезаем padding
        if patch.ndim == 3:
            patch_cropped = patch[:, :h, :w].transpose(
                1, 2, 0)  # C,H,W → H,W,C
        else:
            patch_cropped = patch[:h, :w]

        full_image[y:y+h, x:x+w] += patch_cropped
        count[y:y+h, x:x+w] += 1

    # Усредняем пересекающиеся области
    full_image /= np.maximum(count, 1e-6)

    return full_image

# Датасет для инференса


test_dataset = FundusInferenceDataset(
    image_dir=config.TEST_IMAGE_DIR,
    patch_size=config.PATCH_SIZE,
    stride=config.STRIDE
)
test_loader = DataLoader(
    test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

# Загружаем модели

model_seg, model_skel = load_models(DEVICE)

# Функция апсемплинга


def upsample_to_patch(pred, patch_tensor):
    """
    Апсемплируем предсказание модели до размера patch_tensor
    pred       : [B, C, H, W] или [B, H, W] (добавим C=1, если нужно)
    patch_tensor: [B, C, H, W] (исходный патч)
    """
    if pred.ndim == 3:  # [B,H,W] -> добавляем канал
        pred = pred.unsqueeze(1)

    # size должен быть (H, W)
    H, W = patch_tensor.shape[2], patch_tensor.shape[3]

    return F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)


# INFERENCE

model_seg, model_skel = load_models(DEVICE)


def upsample_to_patch(pred, patch_tensor):
    """
    Апсемплируем предсказание модели до размера patch_tensor
    pred       : [B, C, H, W] или [B, H, W] (добавим C=1, если нужно)
    patch_tensor: [B, C, H, W] (исходный патч)
    """
    if pred.ndim == 3:  # [B,H,W] -> добавляем канал
        pred = pred.unsqueeze(1)

    # size должен быть (H, W)
    H, W = patch_tensor.shape[2], patch_tensor.shape[3]

    return F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)


# Тестирование и сборка
with torch.no_grad():
    # Словарь для хранения патчей каждого изображения
    reconstructed_dict = {}

    for batch in test_loader:
        imgs = batch["image"].to(DEVICE)  # [B,3,H,W]

        # Forward через каскад

        # [B,1,H,W] или [B,C,H,W]
        skel_pred = model_skel(imgs)
        skel_pred_up = upsample_to_patch(skel_pred, imgs)   # до размера патча

        seg_pred = model_seg(imgs, skel_pred_up)            # [B,1,H,W]
        seg_pred_up = upsample_to_patch(seg_pred, imgs)
        seg_pred_sigmoid = torch.sigmoid(seg_pred_up)

        # Сохраняем патчи для сборки

        for i in range(imgs.size(0)):
            img_id = batch["image_id"][i]
            coords = batch["coords"][i].cpu().tolist()
            patch_shape = batch["patch_shape"][i].cpu().tolist()
            full_size = batch["full_size"][i].cpu().tolist()
            patch_pred = seg_pred_sigmoid[i, 0].cpu()  # [H,W]

            if img_id not in reconstructed_dict:
                reconstructed_dict[img_id] = {
                    "patches": [],
                    "coords": [],
                    "patch_shapes": [],
                    "full_size": full_size
                }

            reconstructed_dict[img_id]["patches"].append(patch_pred)
            reconstructed_dict[img_id]["coords"].append(coords)
            reconstructed_dict[img_id]["patch_shapes"].append(patch_shape)

# Собираем полные изображения

for img_id, data in reconstructed_dict.items():
    full_pred = reconstruct_image_from_patches(
        data["patches"],
        data["coords"],
        data["patch_shapes"],
        data["full_size"]
    )

    # Опционально: сохраняем маску
    save_path = os.path.join(
        config.SAVE_DIR_PREDICTION_MASK, f"{img_id}_pred.png")
    plt.imsave(save_path, full_pred, cmap="gray")

"""Dataloader for loading images"""

import os
import random
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from skimage.morphology import skeletonize


def load_image(path):
    """
    Load RGB fundus image.

    EXPECTED:
    - shape: (H, W, 3)
    - dtype: float32
    - range: [0, 1]
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"[ERROR] Cannot read image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    # DEBUG CHECK
    if img.ndim != 3 or img.shape[2] != 3:
        print(f"[WARN] Image shape unexpected: {img.shape}")

    return img


def load_mask(path):
    """
    Load vessel segmentation mask and convert to binary.

    EXPECTED:
    - shape: (H, W)
    - values: {0, 1}
    """
    mask = cv2.imread(path)
    if mask is None:
        raise ValueError(f"[ERROR] Cannot read mask: {path}")

    # Convert color GT -> grayscale
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Any non-zero pixel is vessel
    mask = (mask > 0).astype(np.uint8)

    # DEBUG CHECK
    unique_vals = np.unique(mask)
    if not set(unique_vals).issubset({0, 1}):
        print(f"[WARN] Mask not binary, values: {unique_vals}")

    return mask


def compute_skeleton(mask):
    """
    Compute skeleton from binary vessel mask.

    EXPECTED:
    - shape: (H, W)
    - very sparse binary image
    """
    skel = skeletonize(mask > 0)
    skel = skel.astype(np.uint8)

    # DEBUG CHECK
    if skel.sum() == 0:
        print("[WARN] Skeleton is empty — check mask quality")

    return skel


class VesselPatchSampler:
    def __init__(
        self,
        patch_size=512,
        min_vessel_ratio=0.01,
        max_tries=20
    ):
        """
        patch_size: size of square patch
        min_vessel_ratio: minimum vessel pixels ratio
        """
        self.patch_size = patch_size
        self.min_vessel_ratio = min_vessel_ratio
        self.max_tries = max_tries

    def sample(self, mask):
        """
        Try to sample a patch that contains vessels.

        EXPECTED:
        - Most patches contain visible vessels
        """
        H, W = mask.shape
        ps = self.patch_size

        assert H >= ps and W >= ps, \
            f"[ERROR] Patch size {ps} larger than image {H}x{W}"

        for attempt in range(self.max_tries):
            y = random.randint(0, H - ps)
            x = random.randint(0, W - ps)

            patch = mask[y:y+ps, x:x+ps]
            vessel_ratio = patch.mean()

            if vessel_ratio >= self.min_vessel_ratio:
                return y, x

        # Fallback (should be rare)
        print("[INFO] Fallback random patch used")
        return random.randint(0, H - ps), random.randint(0, W - ps)

# Dataset


class FundusPatchDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        image_ids=None,
        patch_size=512,
        min_vessel_ratio=0.01,
        augment=False,
        debug=False
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.augment = augment
        self.debug = debug

        if image_ids is None:
            self.image_files = sorted(os.listdir(image_dir))
        else:
            self.image_files = sorted(image_ids)
        self.mask_files = self.image_files

        assert len(self.image_files) > 0, "[ERROR] Empty dataset!"

        self.sampler = VesselPatchSampler(
            patch_size=patch_size,
            min_vessel_ratio=min_vessel_ratio
        )

        if self.debug:
            print(f"[INFO] Dataset initialized")
            print(f"  Images: {len(self.image_files)}")
            print(f"  Patch size: {patch_size}")
            print(f" Augment {augment}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load full-resolution data
        img_name = self.image_files[idx]

        img_path = os.path.join(self.image_dir,  img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = load_image(img_path)
        mask = load_mask(mask_path)

        # Sample patch coordinates
        y, x = self.sampler.sample(mask)

        image_patch = image[y:y+self.patch_size, x:x+self.patch_size]
        mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]

        # Skeleton generation
        skel_patch = compute_skeleton(mask_patch)

        # DEBUG PRINT (only once)
        if self.debug and idx == 0:
            print("[DEBUG] Sample inspection:")
            print(f"  Image patch shape: {image_patch.shape}")
            print(f"  Mask sum: {mask_patch.sum()}")
            print(f"  Skeleton sum: {skel_patch.sum()}")

        # Data augmentation (topology-safe)
        if self.augment:
            if random.random() > 0.5:
                image_patch = np.flip(image_patch, axis=1).copy()
                mask_patch = np.flip(mask_patch, axis=1).copy()
                skel_patch = np.flip(skel_patch, axis=1).copy()

            if random.random() > 0.5:
                image_patch = np.flip(image_patch, axis=0).copy()
                mask_patch = np.flip(mask_patch, axis=0).copy()
                skel_patch = np.flip(skel_patch, axis=0).copy()

        # Convert to torch tensors
        image_patch = torch.from_numpy(image_patch).permute(2, 0, 1)
        mask_patch = torch.from_numpy(mask_patch).unsqueeze(0).float()
        skel_patch = torch.from_numpy(skel_patch).unsqueeze(0).float()

        return {
            "image": image_patch,
            "mask": mask_patch,
            "skeleton": skel_patch,
            # уникальный ID
            "image_id": os.path.splitext(img_name)[0],
            "coords": (y, x),
            "full_size": image.shape[:2]  # (H, W)
        }


class FundusInferenceDataset(Dataset):
    def __init__(self, image_dir, patch_size=512, stride=512):
        self.image_dir = image_dir
        self.patch_size = patch_size
        self.stride = stride

        self.image_files = sorted(os.listdir(image_dir))
        self.patches_info = []
        """
        patches_info хранит:
        (image_idx, y, x, H, W)
        """

        # Предварительно считаем ВСЕ координаты патчей
        for image_idx, fname in enumerate(self.image_files):
            img_path = os.path.join(image_dir, fname)
            img = load_image(img_path)
            H, W = img.shape[:2]

            for y in range(0, H, stride):
                for x in range(0, W, stride):
                    self.patches_info.append((image_idx, y, x, H, W))

    def __len__(self):
        return len(self.patches_info)

    def __getitem__(self, idx):
        image_idx, y, x, H, W = self.patches_info[idx]

        img_path = os.path.join(self.image_dir, self.image_files[image_idx])
        image = load_image(img_path)  # [H, W, 3]

        # Вырезаем патч (может быть меньше 512 на краях)
        patch = image[
            y: min(y + self.patch_size, H),
            x: min(x + self.patch_size, W)
        ]

        h, w = patch.shape[:2]  # реальный размер патча

        # Padding до patch_size
        patch_tensor = torch.from_numpy(patch).permute(2, 0, 1)  # [3, h, w]

        pad_h = self.patch_size - h
        pad_w = self.patch_size - w

        if pad_h > 0 or pad_w > 0:
            patch_tensor = torch.nn.functional.pad(
                patch_tensor,
                (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
            )

        # Возвращаем ВСЮ нужную информацию
        return {
            "image": patch_tensor,                  # [3, 512, 512]
            "image_id": self.image_files[image_idx],
            "coords": torch.tensor([y, x]),          # [2]
            "patch_shape": torch.tensor([h, w]),     # [2]
            "full_size": torch.tensor([H, W])        # [2]
        }

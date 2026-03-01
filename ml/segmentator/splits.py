# ml/segmentator/splits.py

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold


def get_image_ids(image_dir, suffixes=(".png", ".jpg", ".jpeg")):
    """
    Возвращает список image_id (имён файлов без пути)
    """
    return sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(suffixes)
    ])


def mask_has_positive(mask_path, threshold=10):
    """
    Возвращает 1 если в маске есть значимое количество пикселей
    """
    mask = np.array(Image.open(mask_path).convert("L"))
    return int(mask.sum() > threshold)


def build_stratify_labels(image_ids, mask_dir):
    """
    Для каждого изображения строим stratify-метку
    """
    labels = []
    for img_id in image_ids:
        mask_path = os.path.join(mask_dir, img_id)
        labels.append(mask_has_positive(mask_path))

    return np.array(labels)


def stratifield_train_val_split(
    image_dir,
    mask_dir,
    val_size=0.2,
    random_state=42
):
    """
    Stratifield train / val split
    """
    image_ids = get_image_ids(image_dir)
    stratify_labels = build_stratify_labels(image_ids, mask_dir)

    train_ids, val_ids = train_test_split(
        image_ids,
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify_labels
    )

    return train_ids, val_ids


def stratifield_kfold_split(
    image_dir,
    mask_dir,
    n_splits=5,
    shuffle=True,
    random_state=42
):
    """
    Генератор k-fold сплитов
    """
    image_ids = get_image_ids(image_dir)
    stratify_labels = build_stratify_labels(image_ids, mask_dir)

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(image_ids, stratify_labels)):
        train_ids = [image_ids[i] for i in train_idx]
        val_ids = [image_ids[i] for i in val_idx]

        yield fold, train_ids, val_ids

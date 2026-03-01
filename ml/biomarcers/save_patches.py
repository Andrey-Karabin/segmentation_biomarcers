
# IMPORTS


import os
import cv2
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


# CONFIG

DATASET_CSV = r"D:\aspirantura\PROF\dataset_final_clean.csv"
AUGMENT_ROOT = r"D:\aspirantura\PROF\augment"
FOLD_TEMPLATE = r"D:\aspirantura\PROF\train_article_fold_{}.csv"
OUTPUT_ROOT = r"D:\aspirantura\PROF\npy_article_fold"

PATCH_SIZE = 512
STRIDE_DEFAULT = 512
STRIDE_RARE = 256

FOLDS = [1, 2, 3]

TARGET_CLASS = "macular_hole"

RARE_CLASSES = [
    "geographic_atrophy",
    "epiretinal_fibrosis",
    "neovascularization",
    "venous_anomalies",
    "laser_coagulates",
    "macular_hole",
]

LABEL_COLUMNS = [
    "hemorrhages", "hard_exudates", "microaneurysms", "drusen",
    "soft_exudates", "edema", "fibrosis", "subretinal_hemorrhage",
    "OD", "geographic_atrophy", "epiretinal_fibrosis",
    "neovascularization", "venous_anomalies",
    "laser_coagulates", "macular_hole"
]

# CLASS COLOR MAP

CLASSES = {
    "ERROR": (211, 255, 5),
    "OD": (250, 250, 55),
    "background": (0, 0, 0),
    "drusen": (115, 71, 30),
    "edema": (109, 230, 213),
    "epiretinal_fibrosis": (88, 4, 46),
    "fibrosis": (196, 67, 237),
    "geographic_atrophy": (250, 189, 124),
    "hard_exudates": (184, 61, 245),
    "hemorrhages": (42, 125, 209),
    "laser_coagulates": (70, 109, 209),
    "macular_hole": (32, 218, 142),
    "microaneurysms": (250, 50, 83),
    "neovascularization": (192, 245, 197),
    "soft_exudates": (61, 245, 61),
    "subretinal_hemorrhage": (98, 243, 161),
    "venous_anomalies": (94, 76, 209),
}

BACKGROUND_CLASSES = ["ERROR", "OD", "background"]
FOREGROUND_CLASSES = [c for c in CLASSES if c not in BACKGROUND_CLASSES]

NUM_CLASSES = len(FOREGROUND_CLASSES) + 1

CLASS_TO_ID = {cls: i + 1 for i, cls in enumerate(FOREGROUND_CLASSES)}
for bg in BACKGROUND_CLASSES:
    CLASS_TO_ID[bg] = 0

COLOR_TO_ID = {CLASSES[c]: CLASS_TO_ID[c] for c in CLASSES}

# UTILS


def binarize_columns(df, columns):
    for col in columns:
        df[col] = (df[col] > 0).astype(int)
    return df


def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def convert_mask_to_id(mask_rgb):
    h, w, _ = mask_rgb.shape
    mask_id = np.zeros((h, w), dtype=np.uint8)

    unique_colors = np.unique(mask_rgb.reshape(-1, 3), axis=0)

    for color in unique_colors:
        color_tuple = tuple(color.tolist())
        if color_tuple in COLOR_TO_ID:
            class_id = COLOR_TO_ID[color_tuple]
            mask_id[np.all(mask_rgb == color, axis=-1)] = class_id
        else:
            print("Unknown color:", color_tuple)

    return mask_id

# TRAIN / TEST SPLIT


def create_train_test_split():

    df = pd.read_csv(DATASET_CSV)
    df = binarize_columns(df, RARE_CLASSES)

    macular_df = df[df[TARGET_CLASS] == 1]
    non_macular_df = df[df[TARGET_CLASS] == 0]

    macular_test = macular_df.sample(
        n=min(4, len(macular_df)), random_state=42)
    macular_train = macular_df.drop(macular_test.index)

    train_df, test_df = train_test_split(
        non_macular_df,
        test_size=0.2,
        random_state=42
    )

    train_df = pd.concat([train_df, macular_train])
    test_df = pd.concat([test_df, macular_test])

    train_df.to_csv("train_article.csv", index=False)
    test_df.to_csv("test_article.csv", index=False)

    print("Train:", len(train_df))
    print("Test:", len(test_df))

# AUGMENTATION


def augment_train():

    train_df = pd.read_csv("train_article.csv")
    train_df = binarize_columns(train_df, RARE_CLASSES)

    rare_df = train_df[train_df[RARE_CLASSES].sum(axis=1) > 0]

    image_out = os.path.join(AUGMENT_ROOT, "images")
    mask_out = os.path.join(AUGMENT_ROOT, "masks")

    os.makedirs(image_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    new_rows = []

    for _, row in rare_df.iterrows():

        img = cv2.imread(row["image"])
        mask = cv2.imread(row["mask"])

        base = os.path.splitext(os.path.basename(row["image"]))[0]

        for flip_type, flip_code in [("hflip", 1), ("vflip", 0)]:

            img_flip = cv2.flip(img, flip_code)
            mask_flip = cv2.flip(mask, flip_code)

            img_path = os.path.join(image_out, f"{base}_{flip_type}.jpg")
            mask_path = os.path.join(mask_out, f"{base}_{flip_type}.png")

            cv2.imwrite(img_path, img_flip)
            cv2.imwrite(mask_path, mask_flip)

            new_row = row.copy()
            new_row["image"] = img_path
            new_row["mask"] = mask_path
            new_rows.append(new_row)

    augmented_df = pd.DataFrame(new_rows)
    train_df = pd.concat([train_df, augmented_df], ignore_index=True)
    train_df.to_csv("train_article.csv", index=False)

    print("Augmentation done. New train size:", len(train_df))

# MULTI-LABEL STRATIFIED K-FOLD


def create_folds():

    train_df = pd.read_csv("train_article.csv")
    train_df = binarize_columns(train_df, LABEL_COLUMNS)

    X = train_df.index.values
    y = train_df[LABEL_COLUMNS].values

    mskf = MultilabelStratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    train_df["fold"] = -1

    for fold, (_, val_idx) in enumerate(mskf.split(X, y)):
        train_df.loc[val_idx, "fold"] = fold

    for f in range(3):
        fold_df = train_df[train_df.fold == f]
        fold_df.to_csv(f"train_article_fold_{f+1}.csv", index=False)

    print("Folds created.")

# PATCH EXTRACTION


def save_patches(df_fold, fold_num):

    img_out = os.path.join(OUTPUT_ROOT, f"fold_{fold_num}", "images")
    mask_out = os.path.join(OUTPUT_ROOT, f"fold_{fold_num}", "masks")

    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    new_rows = []

    for _, row in tqdm(df_fold.iterrows(), total=len(df_fold), desc=f"Fold {fold_num}"):

        img = apply_clahe(cv2.imread(row["image"]))
        mask_rgb = cv2.cvtColor(cv2.imread(row["mask"]), cv2.COLOR_BGR2RGB)

        contains_rare = any(row.get(cls, 0) > 0 for cls in RARE_CLASSES)
        stride = STRIDE_RARE if contains_rare else STRIDE_DEFAULT

        h, w, _ = img.shape

        for top in range(0, h - PATCH_SIZE + 1, stride):
            for left in range(0, w - PATCH_SIZE + 1, stride):

                img_patch = img[top:top+PATCH_SIZE, left:left+PATCH_SIZE]
                mask_patch = mask_rgb[top:top+PATCH_SIZE, left:left+PATCH_SIZE]

                mask_id = convert_mask_to_id(mask_patch)

                base = os.path.splitext(os.path.basename(row["image"]))[0]
                name = f"{base}_{top}_{left}.npy"

                img_path = os.path.join(img_out, name)
                mask_path = os.path.join(mask_out, name)

                np.save(img_path, img_patch)
                np.save(mask_path, mask_id)

                new_row = row.copy()
                new_row["image"] = img_path
                new_row["mask"] = mask_path
                new_rows.append(new_row)

    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(
        os.path.join(OUTPUT_ROOT, f"train_article_fold_{fold_num}.csv"),
        index=False
    )


# MAIN

def main():

    create_train_test_split()
    augment_train()
    create_folds()

    for fold in FOLDS:
        df_fold = pd.read_csv(FOLD_TEMPLATE.format(fold))
        save_patches(df_fold, fold)

    print("Pipeline finished successfully.")


if __name__ == "__main__":
    main()

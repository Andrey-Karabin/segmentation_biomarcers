# ml/segmentator/search_skeleton.py

import json
import itertools
import torch
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.segmentator.config import Config
from ml.segmentator.dataloader import FundusPatchDataset
from ml.segmentator.model_skeleton import SegFormerSkeleton, SkeletonLoss
from ml.segmentator.splits import stratifield_kfold_split


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()

# search space
lr_list = config.LR_LIST
alpha_list = config.ALPHA_LIST
beta_list = config.BETA_LIST

"""
| Сценарий | Что происходит                                    |
| -------- | ------------------------------------------------- |
| α > β    | модель учит локальные пиксели, может рвать сосуд  |
| β > α    | модель учит связность, может терять тонкие детали |
| α ≈ β    | компромисс                                        |

"""

epochs = config.SEARCH_EPOCH
batch_size = config.BATCH_SIZE

result_path = config.RESULTS_PATH
os.makedirs(config.PATH_SEARCH, exist_ok=True)
# ml/segmentator/metrics.py


def skeleton_f1(pred, gt, eps=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    return (2 * tp + eps) / (2 * tp + fp + fn + eps)


def run_one_experiment(lr, alpha, beta, train_ids, val_ids):
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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SegFormerSkeleton(backbone=config.SEGFORMER_SKELETON).to(DEVICE)
    criterion = SkeletonLoss(alpha=alpha, beta=beta)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_f1 = 0.0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            img = batch["image"].to(DEVICE)
            gt = batch["skeleton"].to(DEVICE)

            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, gt)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        f1_scores = []

        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(DEVICE)
                gt = batch["skeleton"].to(DEVICE)
                pred = model(img)
                f1_scores.append(skeleton_f1(pred, gt))

        mean_f1 = torch.stack(f1_scores).mean().item()
        best_f1 = max(best_f1, mean_f1)

    return best_f1


def main():
    splits = list(stratifield_kfold_split(
        config.TRAIN_IMAGE_DIR,
        config.TRAIN_MASK_DIR,
        n_splits=5
    ))

    fold, train_ids, val_ids = splits[0]  # только 1 fold

    results = []

    for lr, alpha, beta in itertools.product(lr_list, alpha_list, beta_list):
        print(f"Testing lr={lr}, alpha={alpha}, beta={beta}")

        f1 = run_one_experiment(lr, alpha, beta, train_ids, val_ids)

        results.append({
            "lr": lr,
            "alpha": alpha,
            "beta": beta,
            "f1": f1
        })

        print(f" → F1: {f1:.4f}")

    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    best = max(results, key=lambda x: x["f1"])
    print("\nBest config:", best)


if __name__ == "__main__":
    main()

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
from skimage.morphology import skeletonize

config = Config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models(DEVICE):
    # Skeleton model
    model_skel = SegFormerSkeleton(
        backbone=config.SEGFORMER_SKELETON).to(DEVICE)
    skel_ckpt = torch.load(config.MODEL_SKELETON_BEST, map_location=DEVICE)
    #  берём только state_dict
    model_skel.load_state_dict(skel_ckpt["model_state_dict"])
    model_skel.eval()
    for p in model_skel.parameters():
        p.requires_grad = False

    # Segmentation model
    model_seg = SegFormerSegmentation().to(DEVICE)
    seg_ckpt = torch.load(config.MODEL_SEGFORMER_BEST, map_location=DEVICE)
    #  берём только state_dict
    model_seg.load_state_dict(seg_ckpt["model_state_dict"])
    model_seg.eval()

    return model_seg, model_skel


def dice_score(pred, gt, eps=1e-6):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    return float((2 * tp + eps) / (2 * tp + fp + fn + eps))


def iou_score(pred, gt, eps=1e-6):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    return float((tp + eps) / (tp + fp + fn + eps))


def precision_score(pred, gt, eps=1e-6):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    return float((tp + eps) / (tp + fp + eps))


def recall_score(pred, gt, eps=1e-6):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = (pred & gt).sum()
    fn = (~pred & gt).sum()
    return float((tp + eps) / (tp + fn + eps))


def accuracy_score(pred, gt, eps=1e-6):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = (pred & gt).sum()
    tn = (~pred & ~gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()
    return float((tp + tn + eps) / (tp + tn + fp + fn + eps))


def f1_score(pred, gt, eps=1e-6):
    p = precision_score(pred, gt, eps)
    r = recall_score(pred, gt, eps)
    return float(2 * p * r / (p + r + eps))


def cldice_score(pred, gt, eps=1e-6):
    """
    Compute clDice for skeletons
    pred, gt: numpy arrays, 0/1 or bool
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    skel_pred = skeletonize(pred)
    skel_gt = skeletonize(gt)

    tprec = (skel_pred & gt).sum() / (skel_pred.sum() + eps)
    tsens = (skel_gt & pred).sum() / (skel_gt.sum() + eps)

    return float(2 * tprec * tsens / (tprec + tsens + eps))

import os
import torch
import torch.nn.functional as F
import numpy as np
import base64
import io
from PIL import Image

# Imports for Model 1 & 2
from ml.segmentator.config import Config as Config12
from ml.segmentator.utils import load_models as load_cascade_models

# Imports for Model 3
from transformers import SegformerForSemanticSegmentation

#  CONFIGURATION
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Config for Model 3
CLASS_TO_ID = {
    "drusen": 1, "edema": 2, "epiretinal_fibrosis": 3, "fibrosis": 4,
    "geographic_atrophy": 5, "hard_exudates": 6, "hemorrhages": 7,
    "laser_coagulates": 8, "macular_hole": 9, "microaneurysms": 10,
    "neovascularization": 11, "soft_exudates": 12, "subretinal_hemorrhage": 13,
    "venous_anomalies": 14
}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
CLASSES_RGB = {
    "background": (0, 0, 0),
    "drusen": (115, 71, 30), "edema": (109, 230, 213), "epiretinal_fibrosis": (88, 4, 46),
    "fibrosis": (196, 67, 237), "geographic_atrophy": (250, 189, 124), "hard_exudates": (184, 61, 245),
    "hemorrhages": (42, 125, 209), "laser_coagulates": (70, 109, 209), "macular_hole": (32, 218, 142),
    "microaneurysms": (250, 50, 83), "neovascularization": (192, 245, 197), "soft_exudates": (61, 245, 61),
    "subretinal_hemorrhage": (98, 243, 161), "venous_anomalies": (94, 76, 209),
}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

#  MODEL LOADING

# Load Cascade Models (1 & 2)
model_seg, model_skel = load_cascade_models(DEVICE)

# Load Model 3 (SegFormer)
# Assuming NUM_CLASSES matches your training setup
NUM_CLASSES = len(CLASS_TO_ID) + 1
model_3 = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
).to(DEVICE)

# Load weights for Model 3
weights_path = r"/app/ml/service/inference/data_biomarcers/best_model.pth"
if os.path.exists(weights_path):
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    model_3.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model 3 loaded from epoch {checkpoint.get('epoch', 'unknown')}")
else:
    print("Warning: Model 3 weights not found at specified path.")

model_3.eval()

#  HELPER FUNCTIONS


def normalize_image(img):
    x = np.array(img).astype(np.float32) / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def array_to_base64(img_array):
    """Converts numpy array (RGB) to base64 string."""
    pil_img = Image.fromarray(img_array.astype(np.uint8))
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

#  PREDICTION FUNCTIONS


def predict_cascade_full_image(image_pil):
    """
    Runs Model 1 + Model 2 on an image using sliding window reconstruction.
    Returns: RGB Mask (Numpy Array)
    """
    config = Config12()
    PATCH_SIZE = config.PATCH_SIZE
    STRIDE = config.STRIDE

    orig_w, orig_h = image_pil.size
    # Assuming simple norm for cascade or reuse normalize_image
    img_np = np.array(image_pil).astype(np.float32) / 255.0

    # Padding logic
    new_w = ((orig_w + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
    new_h = ((orig_h + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
    pad_w = new_w - orig_w
    pad_h = new_h - orig_h

    # Pad image (reflect or constant)
    img_pad = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

    H, W, _ = img_pad.shape
    # Accumulator for probabilities (assuming binary output for cascade for visualization)
    full_probs = np.zeros((1, H, W), dtype=np.float32)
    full_counts = np.zeros((H, W), dtype=np.float32)

    with torch.no_grad():
        for top in range(0, H - PATCH_SIZE + 1, STRIDE):
            for left in range(0, W - PATCH_SIZE + 1, STRIDE):
                patch = img_pad[top:top+PATCH_SIZE, left:left+PATCH_SIZE]

                # Preprocess patch (ensure tensor format matches training)
                # Using standard normalization here, adjust if your cascade needs specific prep
                patch_tensor = torch.from_numpy(patch.transpose(
                    2, 0, 1)).unsqueeze(0).to(DEVICE, dtype=torch.float32)

                # Model 1
                skel_pred = model_skel(patch_tensor)
                # Upsample skel_pred if needed to match patch size
                if skel_pred.shape != patch_tensor.shape:
                    skel_pred = F.interpolate(skel_pred, size=(
                        PATCH_SIZE, PATCH_SIZE), mode="bilinear", align_corners=False)

                # Model 2
                seg_pred = model_seg(patch_tensor, skel_pred)
                seg_pred = F.interpolate(seg_pred, size=(
                    PATCH_SIZE, PATCH_SIZE), mode="bilinear", align_corners=False)

                # Get probabilities
                probs = torch.sigmoid(seg_pred)[0, 0].cpu().numpy()  # [H, W]

                full_probs[:, top:top+PATCH_SIZE,
                           left:left+PATCH_SIZE] += probs
                full_counts[top:top+PATCH_SIZE, left:left+PATCH_SIZE] += 1.0

    # Average overlapping areas
    avg_probs = full_probs[0] / np.maximum(full_counts, 1)
    # Threshold to create binary mask
    mask_binary = (avg_probs > 0.5).astype(np.uint8) * 255

    # Crop back to original size
    final_mask = mask_binary[:orig_h, :orig_w]

    # Convert to RGB for visualization (e.g., Green mask)
    mask_rgb = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    mask_rgb[final_mask == 255] = [0, 255, 0]  # Green color for cascade output

    return mask_rgb


def predict_model_3_full_image(image_pil):
    """
    Runs Model 3 (SegFormer) on an image.
    Returns: RGB Mask (Numpy Array)
    """
    PATCH_SIZE = 512
    STRIDE = 256

    orig_w, orig_h = image_pil.size
    img_np = normalize_image(image_pil)

    # Padding
    new_w = ((orig_w + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
    new_h = ((orig_h + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
    pad_w = new_w - orig_w
    pad_h = new_h - orig_h

    img_pad = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

    H, W, _ = img_pad.shape
    full_probs = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)
    full_counts = np.zeros((H, W), dtype=np.float32)

    with torch.no_grad():
        for top in range(0, H - PATCH_SIZE + 1, STRIDE):
            for left in range(0, W - PATCH_SIZE + 1, STRIDE):
                patch = img_pad[top:top+PATCH_SIZE, left:left+PATCH_SIZE]
                patch_tensor = torch.from_numpy(patch.transpose(
                    2, 0, 1)).unsqueeze(0).to(DEVICE, dtype=torch.float32)

                logits = model_3(pixel_values=patch_tensor).logits
                logits = F.interpolate(
                    logits, (PATCH_SIZE, PATCH_SIZE), mode="bilinear", align_corners=False)
                probs = torch.softmax(logits[0], dim=0).cpu().numpy()

                full_probs[:, top:top+PATCH_SIZE,
                           left:left+PATCH_SIZE] += probs
                full_counts[top:top+PATCH_SIZE, left:left+PATCH_SIZE] += 1.0

    avg_probs = full_probs / np.maximum(full_counts[np.newaxis, :, :], 1)
    full_pred = np.argmax(avg_probs, axis=0)

    # Crop
    full_pred = full_pred[:orig_h, :orig_w]

    # Convert to RGB
    mask_rgb = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    for cls_name, idx in CLASS_TO_ID.items():
        mask_rgb[full_pred == idx] = CLASSES_RGB[cls_name]

    return mask_rgb


def predict_and_show_masks(image_pil):
    """
    Main entry point. Runs both pipelines and returns base64 images.
    """
    # Run Cascade Models
    cascade_mask_rgb = predict_cascade_full_image(image_pil)

    # Run Model 3
    model3_mask_rgb = predict_model_3_full_image(image_pil)

    return {
        "cascade_mask": array_to_base64(cascade_mask_rgb),
        "model_3_mask": array_to_base64(model3_mask_rgb)
    }

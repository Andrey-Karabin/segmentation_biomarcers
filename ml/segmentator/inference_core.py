import torch
import torch.nn.functional as F
import base64
import io
import numpy as np
from PIL import Image
from torchvision import transforms

from ml.segmentator.utils import load_models
from ml.segmentator.config import Config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
config = Config()

model_seg, model_skel = load_models(DEVICE)
model_seg.eval()
model_skel.eval()

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])


def upsample_to_input(pred, input_tensor):
    return F.interpolate(
        pred,
        size=(input_tensor.shape[2], input_tensor.shape[3]),
        mode="bilinear",
        align_corners=False
    )


def run_inference(pil_image: Image.Image):
    # Преобразуем в tensor
    img_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        # Skeleton model
        skel_pred = model_skel(img_tensor)
        skel_pred_up = upsample_to_input(skel_pred, img_tensor)

        # Segmentation model
        seg_pred = model_seg(img_tensor, skel_pred_up)
        seg_pred_up = upsample_to_input(seg_pred, img_tensor)
        seg_pred_sigmoid = torch.sigmoid(seg_pred_up)

    # В numpy
    mask = seg_pred_sigmoid[0, 0].cpu().numpy()
    mask = (mask * 255).astype(np.uint8)

    # В PNG в памяти
    pil_mask = Image.fromarray(mask)
    buffer = io.BytesIO()
    pil_mask.save(buffer, format="PNG")

    encoded = base64.b64encode(buffer.getvalue()).decode()

    return encoded

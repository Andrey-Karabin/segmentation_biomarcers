from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
from ml.service.backend.inference_core import predict_and_show_masks

app = FastAPI(title="Fundus Segmentation API")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives an image, runs inference on Model 1+2 and Model 3.
    Returns base64 encoded masks.
    """
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Run inference
    results = predict_and_show_masks(image)

    return {
        "status": "ok",
        "results": results
    }

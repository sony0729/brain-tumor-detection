"""FastAPI backend: upload an MRI image -> Tumor / No Tumor (VGG16)."""
import io
import os

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "model", "brain_tumor_vgg16.h5"))
CLASSES = ["No Tumor", "Tumor"]  # LabelBinarizer order: ['no', 'yes']

app = FastAPI(title="Brain Tumor Detection API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


def get_model():
    """Load the Keras model once, on first use."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(503, f"Model file not found at {MODEL_PATH}. Run train.py first.")
        from tensorflow.keras.models import load_model

        _model = load_model(MODEL_PATH)
    return _model


@app.get("/")
def root():
    return {"status": "ok", "model_loaded": _model is not None, "model_file_exists": os.path.exists(MODEL_PATH)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "Please upload an image file (jpg/png).")
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB").resize((224, 224))
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Could not read that image.")

    x = np.asarray(img, dtype="float32")[None] / 255.0  # same preprocessing as training
    probs = get_model().predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "prediction": CLASSES[idx],
        "has_tumor": idx == 1,
        "confidence": round(float(probs[idx]), 4),
        "probabilities": {"No Tumor": round(float(probs[0]), 4), "Tumor": round(float(probs[1]), 4)},
    }

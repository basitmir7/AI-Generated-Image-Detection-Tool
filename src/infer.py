"""
Two ways to use this:

1. CLI, single image:
    python src/infer.py --checkpoint checkpoints/best.pt --image path/to/image.jpg

2. API server (for the web app frontend):
    uvicorn infer:app --reload --port 8000
    Then POST an image file (multipart/form-data, field name "file") to
    http://localhost:8000/predict
"""

import argparse
import io

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

from model import build_model
from train import get_device
from dataset import IMAGENET_MEAN, IMAGENET_STD

CHECKPOINT_PATH = "checkpoints/best.pt"
BACKBONE = "resnet50"
IMAGE_SIZE = 224

_device = get_device()
_model = None


def _get_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def load_model(checkpoint_path: str = CHECKPOINT_PATH, backbone: str = BACKBONE):
    global _model
    model = build_model(backbone, pretrained=False).to(_device)
    ckpt = torch.load(checkpoint_path, map_location=_device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    _model = model
    return model


def predict_image(image: np.ndarray) -> dict:
    """image: RGB numpy array (H, W, 3). Returns probability of being AI-generated."""
    global _model
    if _model is None:
        load_model()

    transform = _get_transform()
    tensor = transform(image=image)["image"].unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    p_fake = probs[1].item()
    return {
        "probability_ai_generated": round(p_fake, 4),
        "label": "ai_generated" if p_fake >= 0.5 else "real",
        "confidence": round(max(p_fake, 1 - p_fake), 4),
    }


# ---- CLI ----
def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--backbone", type=str, default=BACKBONE)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    load_model(args.checkpoint, args.backbone)
    image = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    result = predict_image(image)
    print(result)


# ---- FastAPI app ----
try:
    from fastapi import FastAPI, File, UploadFile
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="AI Image Detector")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten this before deploying publicly
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    def _startup():
        load_model()

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = np.array(pil_image)
        result = predict_image(image)
        result["note"] = (
            "This is a probabilistic estimate, not a certainty. "
            "Detection accuracy varies across image generators and image editing."
        )
        return result

    @app.get("/health")
    def health():
        return {"status": "ok", "device": str(_device)}

except ImportError:
    app = None  # fastapi not installed -- CLI mode still works


if __name__ == "__main__":
    _cli()

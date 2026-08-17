# AI Image Detector

Binary classifier that estimates whether an image is real or AI-generated.
Built to fine-tune a pretrained CNN backbone on Apple Silicon (MPS), with
checkpointing after every epoch so interruptions never cost more than the
current epoch's progress.

## 1. Environment setup

```bash
cd ai-image-detector
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Verify MPS is available:
```bash
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```
Should print `True`. If it prints `False`, update PyTorch (`pip install --upgrade torch torchvision`) and make sure you're on macOS 12.3+.

## 2. Get a dataset

Start small to validate the pipeline, then scale up.

**Step A — CIFAKE (fastest way to get something working)**
- ~120k images (60k real, 60k AI-generated via Stable Diffusion), 32x32 resolution.
- Download from Kaggle: search "CIFAKE" (requires a free Kaggle account).
- Low resolution means fast training, but the model won't generalize well to
  other generators or full-resolution images — treat this as a pipeline
  sanity check, not your final model.

**Step B — a multi-generator, higher-resolution dataset (for the real version)**
- Look at **GenImage** or **ArtiFact** — both include outputs from multiple
  generators (Midjourney, SDXL, DALL-E, etc.), which is what makes the model
  generalize instead of just memorizing one generator's noise fingerprint.
- These are much larger (tens of GB) — check your disk space first.

**Folder layout this project expects** (create this yourself after downloading):
```
data/
  train/
    real/   *.jpg / *.png
    fake/   *.jpg / *.png
  val/
    real/
    fake/
  test/
    real/
    fake/
```
An 80/10/10 train/val/test split is a reasonable default.

## 3. Train

```bash
cd src
python3 train.py --data-dir ../data --backbone resnet50 --epochs 15 --batch-size 32
```

The script auto-detects your device (cuda → mps → cpu), so this exact
command will also work unchanged if you ever do move training to a CUDA
machine later.

**Checkpointing is automatic:**
- `checkpoints/last.pt` is overwritten after every epoch.
- `checkpoints/best.pt` is saved whenever validation loss improves.
- If training is interrupted (closed laptop, crash, anything), resume with:
```bash
python3 train.py --data-dir ../data --backbone resnet50 --epochs 15 --resume
```

**Realistic M2 expectations:** roughly 2-4x slower than an RTX 3070 for the
same job. Start with a smaller subset of your data (a few thousand images
per class) to sanity-check the full pipeline runs end-to-end before
committing to a multi-hour run on the full dataset.

Track progress live with TensorBoard:
```bash
tensorboard --logdir runs
```

## 4. Evaluate

```bash
python3 evaluate.py --checkpoint ../checkpoints/best.pt --data-dir ../data --split test
```

Reports accuracy, precision/recall, confusion matrix, false positive rate,
and ROC-AUC. To test generalization, point `--data-dir` at a separate test
folder containing images from a generator that wasn't in your training set
(same `test/real`, `test/fake` structure) — this is the most interesting
result to report and write up.

## 5. Run inference

Single image via CLI:
```bash
python3 infer.py --checkpoint ../checkpoints/best.pt --image path/to/image.jpg
```

As an API (for a frontend to call):
```bash
uvicorn infer:app --reload --port 8000
```
Then `POST` a file to `http://localhost:8000/predict` (multipart form field `file`).
Response:
```json
{
  "probability_ai_generated": 0.87,
  "label": "ai_generated",
  "confidence": 0.87,
  "note": "This is a probabilistic estimate, not a certainty. ..."
}
```

## Notes on framing this honestly

- Report the model's false positive rate prominently — wrongly labeling a
  real photo as AI-generated is the more damaging failure mode for
  harassment/misinformation use cases.
- Detectors degrade against generators they weren't trained on. Say so in
  your UI and README rather than presenting the tool as infallible.
- A probability score with a confidence caveat is more honest (and more
  useful) than a hard yes/no.

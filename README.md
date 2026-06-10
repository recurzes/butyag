# Butyag Server — FastAPI + ONNX Inference

Upload a chest X-ray → get prediction + confidence score + Grad-CAM overlay.

---

## Project Structure

```
butyag-server/
├── app/
│   ├── main.py        # FastAPI app, /predict endpoint
│   ├── inference.py   # ONNX session, preprocessing, Grad-CAM
│   └── config.py      # Shared constants (IMAGE_SIZE, threshold, etc.)
├── frontend/
│   └── src/App.jsx    # React upload UI
├── exports/           # Place butyag.onnx + butyag_best.pth here
├── export_onnx.py     # Run once after training to generate butyag.onnx
├── Dockerfile
└── requirements.txt
```

---

## Step 1 — Export ONNX (run once after training)

```bash
# From butyag-server/
uv run python export_onnx.py
# Outputs: exports/butyag.onnx
```

Also copy your trained weights:
```bash
cp ../butyag/outputs/butyag_best.pth exports/
```

---

## Step 2 — Run locally

**Backend:**
```bash
uv sync                          # installs all deps from pyproject.toml
uv run uvicorn app.main:app --reload
# API running at http://localhost:8000
```

**Frontend (separate terminal):**
```bash
cd frontend
npm install
npm run dev
# UI at http://localhost:5173
```

---

## Step 3 — Update threshold

After running `evaluate.py` from the training project, update `THRESHOLD` in `app/config.py`
with the Youden's J value printed to console.

---

## Step 4 — Deploy (Docker + Dokploy)

```bash
# Build
docker build -t butyag-server .

# Run
docker run -p 8000:8000 butyag-server
```

Add to Dokploy as a Docker app — set port 8000, mount `exports/` as a volume
if you want to swap model weights without rebuilding the image.

---

## API

```
POST /predict
Content-Type: multipart/form-data
Body: file=<image>

Response:
{
  "prediction":     "PNEUMONIA",
  "confidence":     0.874,
  "pneumonia_prob": 0.874,
  "normal_prob":    0.126,
  "threshold":      0.47,
  "gradcam_base64": "<base64 PNG or null>"
}
```

---

## Notes

- ONNX Runtime loads once at startup, reused per request — no cold start per call
- Grad-CAM uses a separate PyTorch session loaded lazily on first request
- If `butyag_best.pth` is absent, Grad-CAM is skipped and `gradcam_base64` returns null
- Frontend served from `/` in production (React build copied into FastAPI StaticFiles)
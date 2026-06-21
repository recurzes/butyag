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
# UI at http://localhost:5173 — /predict is proxied to the backend on :8000
```

---

## Step 3 — Update threshold

After running `evaluate.py` from the training project, update `THRESHOLD` in `app/config.py`
with the Youden's J value printed to console.

---

## Step 4 — Deploy (Docker Compose + Dokploy)

### Generate model artifacts (once)

```bash
cd training
uv run python export_onnx.py
mv exports/butyag.onnx ../exports/
cp ../outputs/butyag_best.pth ../exports/
```

### Local Docker Compose

```bash
docker compose up -d --build
```

Assign a domain to the `frontend` service (container port **80**). The frontend nginx proxies `/predict` to the internal backend over `dokploy-network`.

**Ports (production vs dev):**

| Context | Port | Notes |
|---------|------|-------|
| Local Vite dev | **5173** | `npm run dev` only — not used in Docker |
| Docker frontend (container) | **80** | nginx inside the image |
| Docker frontend (host) | **5175** | `ports: "5175:80"` in compose — for Cloudflare Tunnel |
| Backend (internal) | **8080** | Not published; reached as `backend:8080` from frontend |

**Cloudflare Tunnel** (point tunnel at host port 5175):

```yaml
ingress:
  - hostname: your-domain.example.com
    service: http://127.0.0.1:5175
  - service: http_status:404
```

Use Cloudflare Tunnel **or** Dokploy Domains — not both on the same hostname (avoids double-proxy 502s).

### Dokploy

1. Push repo with `docker-compose.yml`
2. New Project → Compose → **Docker Compose** (not Stack)
3. Compose path: `docker-compose.yml`
4. **Domains tab** (if not using Cloudflare Tunnel) — add domain for service `frontend`, container port **80**, enable HTTPS
5. Do **not** expose `backend` publicly
6. Deploy

Environment variables (optional, set in Dokploy UI):

| Variable | Purpose |
|----------|---------|
| `VITE_API_URL` | Leave empty in production (same-origin proxy) |
| `CORS_ORIGINS` | Comma-separated origins for local dev |

See [`.env.example`](.env.example) for defaults.

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
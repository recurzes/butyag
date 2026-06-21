# Model artifacts for deployment

Place `butyag.onnx` here before building the backend image.

Generate from trained weights:

```bash
cd training
uv run python export_onnx.py
# copies output to training/exports/butyag.onnx — move to repo root exports/
mv exports/butyag.onnx ../exports/
cp ../outputs/butyag_best.pth ../exports/
```

Alternatively, upload via Dokploy File Mounts (Advanced → Mounts) to `../files/exports/` and bind-mount into the backend container.

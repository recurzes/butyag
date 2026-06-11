from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import (
    ONNX_MODEL_PATH, PTH_MODEL_PATH,
    MAX_FILE_SIZE_MB, ALLOWED_MIMETYPES
)
from app.inference import OnnxInferenceSession, preprocess_image, generate_gradcam


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Butyag server...")
    app.state.ort_session = OnnxInferenceSession(ONNX_MODEL_PATH)
    print("Server ready.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="Butyag - Chest X-Ray Screening API",
    description="EfficientNet-B3 inference server for RHU chest X-ray screening",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


class PredictResponse(BaseModel):
    prediction:         str
    confidence:         float
    pneumonia_prob:     float
    normal_prob:        float
    threshold:          float
    gradcam_base64:     str | None


# Routes
@app.get("/")
def health():
    return {"status": "ok", "model": "EfficientNet-B3 (ONNX)", "version": "1.0.0"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_MIMETYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Upload a JPEG or PNG"
        )

    image_bytes = await file.read()
    size_mb = len(image_bytes) / 1e6
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=443,
            detail=f"File too large({size_mb:.1f} MB). Maximum is {MAX_FILE_SIZE_MB} MB."
        )

    try:
        input_array, original_img = preprocess_image(image_bytes)
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Could not read image. Make sure it is a valid JPEG or PNG"
        )

    session = request.app.state.ort_session
    result = session.predict(input_array)

    gradcam_b64 = None
    pth_path = Path(PTH_MODEL_PATH)
    if pth_path.exists():
        gradcam_b64 = generate_gradcam(image_bytes, str(pth_path), original_img)

    return PredictResponse(**result, gradcam_base64=gradcam_b64)


# Serve React Build (comment on dev)
# frontend_dist = Path("frontend/dist")
# if frontend_dist.exists():
#     app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")
#
#
#     @app.get("/{full_path:path}")
#     def serve_frontend(full_path: str):
#         return FileResponse(str(frontend_dist / "index.html"))

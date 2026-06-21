import base64
import io
import numpy as np
import torch
from PIL import Image

import onnxruntime as ort
from torchvision import transforms

from app.config import IMAGE_SIZE, MEAN, STD, CLASSES, THRESHOLD


# Preprocessing
_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


def preprocess_image(image_bytes: bytes) -> tuple[np.ndarray, Image.Image]:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _transform(img).unsqueeze(0)
    return tensor.numpy().astype(np.float32), img


# ONNX Inference
class OnnxInferenceSession:
    def __init__(self, model_path: str):
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"ONNX session loaded: {model_path}")
        print(f"  Input: {self.input_name}  {self.session.get_inputs()[0].shape}")
        print(f"  Output: {self.session.get_outputs()[0].name}")

    def predict(self, input_array: np.ndarray) -> dict:
        logit = self.session.run(None, {self.input_name: input_array})[0]

        pneumonia_prob = float(1 / (1 + np.exp(-logit[0][0])))
        normal_prob = 1.0 - pneumonia_prob

        prediction = CLASSES[1] if pneumonia_prob >= THRESHOLD else CLASSES[0]
        confidence = pneumonia_prob if prediction == "PNEUMONIA" else normal_prob

        return {
            "prediction":       prediction,
            "confidence":       round(confidence, 4),
            "pneumonia_prob":   round(pneumonia_prob, 4),
            "normal_prob":      round(normal_prob, 4),
            "threshold":        THRESHOLD
        }


_gradcam_model = None

def _get_gradcam_model(pth_path: str):
    global _gradcam_model
    if _gradcam_model is None:
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from training.model import build_model
        from training.predict import GradCAM

        model = build_model(device="cpu", freeze_backbone=False)
        model.load_state_dict(torch.load(pth_path, map_location="cpu"))
        model.eval()
        _gradcam_model = (model, GradCAM(model))
        print("Grad-CAM PyTorch model loaded (lazy init)")

    return _gradcam_model


def generate_gradcam(
        image_bytes: bytes,
        pth_path: str,
        original_img: Image.Image
) -> str:
    try:
        import torch
        import matplotlib.cm as cm

        model, grad_cam = _get_gradcam_model(pth_path)

        tensor = _transform(original_img.resize((IMAGE_SIZE, IMAGE_SIZE)))
        tensor = tensor.unsqueeze(0)

        cam = grad_cam.generate(tensor)

        img_array = np.array(original_img.resize((IMAGE_SIZE, IMAGE_SIZE))) / 255.0
        cam_img = Image.fromarray((cam * 255).astype(np.uint8))
        cam_img = cam_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        cam_array = np.array(cam_img) / 255.0
        heatmap = cm.jet(cam_array)[:, :, :3]
        overlay = np.clip((1 - 0.4) * img_array + 0.4 * heatmap, 0, 1)

        overlay_pil = Image.fromarray((overlay * 255).astype(np.uint8))
        buf = io.BytesIO()
        overlay_pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    except Exception as e:
        print(f"Grad-CAM failed (non-critical): {e}")
        return None
import sys
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from sympy.physics import pring
from torch.autograd import backward
from torchvision import transforms

from training.dataset import IMAGE_SIZE, MEAN, STD, CLASSES
from training.model import build_model


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        target_layer = self.model.backbone[-1]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model.eval()
        input_tensor.requires_grad_(True)

        logit = self.model(input_tensor)
        self.model.zero_grad()
        logit.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam


# Preprocessing
def preprocess(image_path: str, device: str) -> tuple:
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    return img, tensor


# Overlay
def overlay_cam(original_img: Image.Image, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    img_array = np.array(original_img.resize((IMAGE_SIZE, IMAGE_SIZE))) / 255.0

    cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR
    )
    cam_array = np.array(cam_resized) / 255.0

    heatmap = cm.jet(cam_array)[:, :, :3]
    overlay = (1 - alpha) * img_array + alpha * heatmap
    return np.clip(overlay, 0, 1)


# Main Inference
def predict(image_path: str, model_path: str = "../outputs/butyag_best.pth", threshold: float = 0.5,
            output_dir: str = "../outputs"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(device=device, freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location=device))

    grad_cam = GradCAM(model)

    original_img, tensor = preprocess(image_path, device)
    logit = model(tensor)
    prob = torch.sigmoid(logit).item()
    pred_class = CLASSES[1] if prob >= threshold else CLASSES[0]
    confidence = prob if prob >= threshold else 1 - prob

    cam = grad_cam.generate(tensor)
    overlay = overlay_cam(original_img, cam)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_img.resize((IMAGE_SIZE, IMAGE_SIZE)), cmap="gray")
    axes[0].set_title("Original X-Ray", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM Activation", fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046)

    axes[2].imshow(cam, cmap="jet")
    axes[2].set_title("Overlay (CAM on X-Ray)", fontweight="bold")
    axes[2].axis("off")

    result_color = "#F44336" if pred_class == "PNEUMONIA" else "#4CAF50"
    fig.suptitle(
        f"Prediction: {pred_class}   |   Probability: {prob:.4f}   |   Confidence: {confidence:.2%}",
        fontsize=14, fontweight="bold", color=result_color
    )

    save_path = out_dir / f"gradcam_{Path(image_path).stem}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n{'=' * 50}")
    print(f"  File:        {image_path}")
    print(f"  Prediction:  {pred_class}")
    print(f"  Probability: {prob:.4f}")
    print(f"  Confidence:  {confidence:.2%}")
    print(f"  Threshold:   {threshold}")
    print(f"  Grad-CAM:    {save_path}")
    plt.show()

    return {"prediction": pred_class, "probability": prob, "confidence": confidence}


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_xray_image>")
        print("Example: python predict.py data/test/PNEUMONIA/person1_virus_1.jpeg")
        sys.exit(1)

    image_path = sys.argv[1]
    result = predict(image_path)
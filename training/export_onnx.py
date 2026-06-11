import sys
import torch
import onnx
from pathlib import Path

from app.config import IMAGE_SIZE

sys.path.insert(0, "../butyag/src")
from training.model import build_model

PTH_PATH = "../butyag/outputs/butyag_best.pth"
ONNX_PATH = "exports/butyag.onnx"


def export():
    Path("exports").mkdir(exist_ok=True)

    device = "cpu"
    print("Loading trained model...")
    model = build_model(device=device, freeze_backbone=False)
    model.load_state_dict(torch.load(PTH_PATH, map_location=device))
    model.eval()

    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["xray_image"],
        output_names=["logit"],
        dynamic_axes={
            "xray_image": {0: "batch_size"},
            "logit": {0: "batch_size"}
        },
        opset_version=18,
        export_params=True,
        do_constant_folding=True
    )

    print("Verifying ONNX model...")
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)

    size_mb = Path(ONNX_PATH).stat().st_size / 1e6
    print("Export successful")
    print(f"  Path:  {ONNX_PATH}")
    print(f"  Size:  {size_mb:.1f} MB")
    print(f"\nInput:  {onnx_model.graph.input[0].name}  "
          f"shape: (batch, 3, {IMAGE_SIZE}, {IMAGE_SIZE})")
    print(f"Output: {onnx_model.graph.output[0].name}  "
          f"shape: (batch, 1)  <- raw logit, apply sigmoid manually")
    print("\nNext step: python -m uvicorn app.main:app --reload")


if __name__ == '__main__':
    export()
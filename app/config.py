ONNX_MODEL_PATH = "exports/butyag.onnx"
PTH_MODEL_PATH = "exports/butyag_best.pth"

IMAGE_SIZE = 300
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

CLASSES = ["NORMAL", "PNEUMONIA"]
THRESHOLD = 0.47

MAX_FILE_SIZE_MB = 10
ALLOWED_MIMETYPES = {"image/jpeg", "image/jpg", "image/png"}

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg")
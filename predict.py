import argparse
import json
import os
from pathlib import Path

import timm
import torch
from PIL import Image
from torchvision import transforms


BASE_DIR = Path(__file__).resolve().parent
EXTERNAL_MODEL_PATH = Path(r"D:\color\best_efficientnet_b0_cat_color.pth")
BUNDLED_MODEL_PATH = BASE_DIR / "best_efficientnet_b0_cat_color.pth"
SETTINGS_PATH = Path(
    os.getenv(
        "APP_SETTINGS_PATH",
        str(Path.home() / ".config" / "cat_color_project" / "runtime_settings.json"),
    )
)
DEFAULT_MODEL = EXTERNAL_MODEL_PATH if EXTERNAL_MODEL_PATH.exists() else BUNDLED_MODEL_PATH
DEFAULT_CONFIDENCE_THRESHOLD = 0.5


def load_model(model_path: Path, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]
    model_name = checkpoint.get("model_name", "efficientnet_b0")
    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, int(checkpoint.get("img_size", 300))


def build_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def load_default_threshold() -> float:
    raw_value = DEFAULT_CONFIDENCE_THRESHOLD

    if SETTINGS_PATH.exists():
        try:
            payload = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        raw_value = payload.get("confidence_threshold", raw_value)

    env_value = os.getenv("CONFIDENCE_THRESHOLD")
    if env_value not in (None, ""):
        raw_value = env_value

    return float(raw_value)


def main():
    parser = argparse.ArgumentParser(description="Predict cat color from an image.")
    parser.add_argument("image", type=Path, help="Path to the image file.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=load_default_threshold())
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, img_size = load_model(args.model, device)
    preprocess = build_transform(img_size)

    image = Image.open(args.image).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        top_k = min(max(args.top_k, 1), len(class_names))
        scores, indices = torch.topk(probabilities, k=top_k)

    scores_cpu = scores.cpu().tolist()
    indices_cpu = indices.cpu().tolist()
    top_score = float(scores_cpu[0])
    top_name = class_names[int(indices_cpu[0])]

    if top_score < args.threshold:
        print(f"Top-1 confidence {top_score:.4f} < {args.threshold:.2f}: 无法判断")
        print(f"Highest candidate: {top_name}")

    for rank, (score, index) in enumerate(zip(scores_cpu, indices_cpu), start=1):
        print(f"{rank}. {class_names[int(index)]}: {float(score):.4f}")


if __name__ == "__main__":
    main()

import json
import os
import ipaddress
import socket
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

import timm
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms


BASE_DIR = Path(__file__).resolve().parent
EXTERNAL_MODEL_PATH = Path(r"D:\color\best_efficientnet_b0_cat_color.pth")
BUNDLED_MODEL_PATH = BASE_DIR / "best_efficientnet_b0_cat_color.pth"
MODEL_PATH = EXTERNAL_MODEL_PATH if EXTERNAL_MODEL_PATH.exists() else BUNDLED_MODEL_PATH
INDEX_HTML_PATH = BASE_DIR / "index.html"
SETTINGS_PATH = Path(
    os.getenv(
        "APP_SETTINGS_PATH",
        str(Path.home() / ".config" / "cat_color_project" / "runtime_settings.json"),
    )
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
UNCERTAIN_CLASS_NAME = "Unknown"
UNCERTAIN_DISPLAY_NAME = "无法判断"
REMOTE_IMAGE_TIMEOUT = 12
REMOTE_IMAGE_MAX_BYTES = 10 * 1024 * 1024

app = FastAPI(title="Cat Color Recognition API")

model = None
class_names = []
preprocess = None
confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD


class ThresholdUpdate(BaseModel):
    confidence_threshold: float


class ImageUrlRequest(BaseModel):
    image_url: str


class ImageUrlBatchRequest(BaseModel):
    image_urls: list[str]


INDEX_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>猫咪颜色识别</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #17212b;
      --muted: #637083;
      --line: #d9e0e7;
      --surface: #ffffff;
      --soft: #f4f7f9;
      --accent: #1f7a6b;
      --accent-dark: #15584d;
      --warn: #a94422;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, #f7faf9 0%, #eef4f7 100%);
      min-height: 100vh;
    }
    .shell {
      width: min(1120px, calc(100% - 32px));
      margin: 0 auto;
      padding: 32px 0;
    }
    header {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 20px;
      margin-bottom: 22px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: clamp(28px, 4vw, 44px);
      line-height: 1.08;
      font-weight: 800;
    }
    .sub {
      margin: 0;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.6;
    }
    .status {
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.72);
      padding: 10px 12px;
      border-radius: 8px;
      min-width: 170px;
      font-size: 13px;
      color: var(--muted);
    }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(320px, 0.95fr);
      gap: 18px;
      align-items: stretch;
    }
    .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 18px 45px rgba(23, 33, 43, 0.08);
      overflow: hidden;
    }
    .upload {
      min-height: 540px;
      display: grid;
      grid-template-rows: 1fr auto;
    }
    .drop {
      margin: 18px;
      border: 1.5px dashed #aab7c4;
      border-radius: 8px;
      background: var(--soft);
      display: grid;
      place-items: center;
      position: relative;
      overflow: hidden;
      min-height: 400px;
      cursor: pointer;
    }
    .drop.drag { border-color: var(--accent); background: #edf8f5; }
    .empty {
      text-align: center;
      padding: 24px;
      color: var(--muted);
    }
    .empty strong {
      display: block;
      color: var(--ink);
      font-size: 21px;
      margin-bottom: 8px;
    }
    #preview {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: none;
      background: #111820;
    }
    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      border-top: 1px solid var(--line);
      padding: 14px 18px;
    }
    input[type="file"] { display: none; }
    button, .fileButton {
      border: 0;
      border-radius: 7px;
      min-height: 42px;
      padding: 0 16px;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }
    .fileButton {
      background: #e8eef2;
      color: var(--ink);
    }
    button {
      background: var(--accent);
      color: white;
    }
    button:hover { background: var(--accent-dark); }
    button:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }
    .result {
      padding: 22px;
      display: flex;
      flex-direction: column;
      min-height: 540px;
    }
    .result h2 {
      margin: 0 0 18px;
      font-size: 22px;
    }
    .winner {
      border: 1px solid var(--line);
      background: #f8fbfa;
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 18px;
    }
    .winner .label {
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }
    .winner .name {
      font-size: 26px;
      font-weight: 800;
      line-height: 1.2;
      overflow-wrap: anywhere;
    }
    .winner .conf {
      margin-top: 10px;
      color: var(--accent-dark);
      font-weight: 700;
    }
    .bars {
      display: grid;
      gap: 12px;
    }
    .barRow {
      display: grid;
      gap: 6px;
    }
    .barTop {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 14px;
    }
    .barTrack {
      height: 11px;
      border-radius: 999px;
      background: #e6edf1;
      overflow: hidden;
    }
    .barFill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #1f7a6b, #d39b32);
      transition: width 260ms ease;
    }
    .message {
      margin-top: auto;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.6;
    }
    .error {
      color: var(--warn);
      font-weight: 700;
    }
    @media (max-width: 780px) {
      .shell { width: min(100% - 22px, 1120px); padding: 18px 0; }
      header { align-items: stretch; flex-direction: column; }
      main { grid-template-columns: 1fr; }
      .upload, .result { min-height: auto; }
      .drop { min-height: 310px; margin: 12px; }
      .actions { padding: 12px; }
      button, .fileButton { flex: 1 1 150px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <div>
        <h1>猫咪颜色识别</h1>
        <p class="sub">上传猫咪图片，模型会返回最可能的毛色类别和 Top 3 置信度。</p>
      </div>
      <div class="status" id="status">模型状态检测中...</div>
    </header>

    <main>
      <section class="panel upload">
        <div class="drop" id="drop" role="button" tabindex="0" aria-label="上传图片">
          <img id="preview" alt="待识别图片预览">
          <div class="empty" id="empty">
            <strong>点击或拖入一张猫咪图片</strong>
            JPG、PNG、WEBP 均可
          </div>
        </div>
        <div class="actions">
          <button id="choose" type="button" class="fileButton">上传图片</button>
          <input id="file" type="file" accept="image/*">
          <button id="predict" disabled>开始识别</button>
        </div>
      </section>

      <section class="panel result">
        <h2>识别结果</h2>
        <div class="winner">
          <div class="label">最可能类别</div>
          <div class="name" id="winnerName">等待上传</div>
          <div class="conf" id="winnerConf">置信度 --</div>
        </div>
        <div class="bars" id="bars"></div>
        <p class="message" id="message">结果会显示模型预测类别，例如 Black_cat、Orange_Cat、White_cat 等。</p>
      </section>
    </main>
  </div>

  <script>
    const fileInput = document.querySelector("#file");
    const chooseButton = document.querySelector("#choose");
    const predictButton = document.querySelector("#predict");
    const preview = document.querySelector("#preview");
    const empty = document.querySelector("#empty");
    const drop = document.querySelector("#drop");
    const winnerName = document.querySelector("#winnerName");
    const winnerConf = document.querySelector("#winnerConf");
    const bars = document.querySelector("#bars");
    const message = document.querySelector("#message");
    const statusBox = document.querySelector("#status");
    let selectedFile = null;

    function formatPct(value) {
      return `${(value * 100).toFixed(2)}%`;
    }

    function setFile(file) {
      if (!file || !file.type.startsWith("image/")) {
        message.innerHTML = "<span class='error'>请选择图片文件。</span>";
        return;
      }
      selectedFile = file;
      preview.src = URL.createObjectURL(file);
      preview.style.display = "block";
      empty.style.display = "none";
      predictButton.disabled = false;
      winnerName.textContent = "等待识别";
      winnerConf.textContent = "置信度 --";
      bars.innerHTML = "";
      message.textContent = `已选择：${file.name}`;
    }

    function openPicker() {
      fileInput.value = "";
      fileInput.click();
    }

    async function loadHealth() {
      try {
        const res = await fetch("/health");
        const data = await res.json();
        statusBox.textContent = `模型已加载 | ${data.classes} 类 | ${data.device}`;
      } catch {
        statusBox.textContent = "模型状态暂不可用";
      }
    }

    async function predict() {
      if (!selectedFile) return;
      predictButton.disabled = true;
      predictButton.textContent = "识别中...";
      message.textContent = "正在分析图片，请稍等。";

      const form = new FormData();
      form.append("file", selectedFile);

      try {
        const res = await fetch("/predict?top_k=3", { method: "POST", body: form });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || "识别失败");
        }
        const data = await res.json();
        winnerName.textContent = data.prediction.class_name;
        winnerConf.textContent = `置信度 ${formatPct(data.prediction.confidence)}`;
        bars.innerHTML = data.top_k.map(item => {
          const pct = Math.max(0, Math.min(100, item.confidence * 100));
          return `
            <div class="barRow">
              <div class="barTop">
                <strong>${item.class_name}</strong>
                <span>${formatPct(item.confidence)}</span>
              </div>
              <div class="barTrack"><div class="barFill" style="width:${pct}%"></div></div>
            </div>
          `;
        }).join("");
        message.textContent = `文件：${data.filename}`;
      } catch (err) {
        message.innerHTML = `<span class="error">${err.message}</span>`;
      } finally {
        predictButton.disabled = false;
        predictButton.textContent = "开始识别";
      }
    }

    fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
    chooseButton.addEventListener("click", openPicker);
    drop.addEventListener("click", openPicker);
    drop.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openPicker();
      }
    });
    predictButton.addEventListener("click", predict);
    ["dragenter", "dragover"].forEach(name => {
      drop.addEventListener(name, event => {
        event.preventDefault();
        drop.classList.add("drag");
      });
    });
    ["dragleave", "drop"].forEach(name => {
      drop.addEventListener(name, event => {
        event.preventDefault();
        drop.classList.remove("drag");
      });
    });
    drop.addEventListener("drop", event => setFile(event.dataTransfer.files[0]));
    loadHealth();
  </script>
</body>
</html>
"""


def load_checkpoint():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return torch.load(MODEL_PATH, map_location=DEVICE)


def normalize_threshold(value) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence_threshold must be a number") from exc

    if threshold < 0 or threshold > 1:
        raise ValueError("confidence_threshold must be between 0 and 1")

    return threshold


def load_confidence_threshold() -> float:
    raw_value = DEFAULT_CONFIDENCE_THRESHOLD

    if SETTINGS_PATH.exists():
        try:
            payload = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid settings file: {SETTINGS_PATH}") from exc
        raw_value = payload.get("confidence_threshold", raw_value)

    env_value = os.getenv("CONFIDENCE_THRESHOLD")
    if env_value not in (None, ""):
        raw_value = env_value

    return normalize_threshold(raw_value)


def save_confidence_threshold(value: float) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(
        json.dumps({"confidence_threshold": value}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def is_public_ip(hostname: str) -> bool:
    try:
        ip = ipaddress.ip_address(hostname)
        return ip.is_global
    except ValueError:
        pass

    try:
        addresses = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise HTTPException(status_code=400, detail="Image URL hostname could not be resolved") from exc

    resolved_ips = set()
    for entry in addresses:
        sockaddr = entry[4]
        if not sockaddr:
            continue
        resolved_ips.add(sockaddr[0])

    if not resolved_ips:
        raise HTTPException(status_code=400, detail="Image URL hostname could not be resolved")

    for raw_ip in resolved_ips:
        try:
            if not ipaddress.ip_address(raw_ip).is_global:
                return False
        except ValueError:
            return False

    return True


def fetch_remote_image(image_url: str) -> tuple[Image.Image, str]:
    parsed = urlparse(image_url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="Image URL must start with http:// or https://")

    if not parsed.netloc or not parsed.hostname:
        raise HTTPException(status_code=400, detail="Image URL is invalid")

    if parsed.username or parsed.password:
        raise HTTPException(status_code=400, detail="Image URL cannot include username or password")

    if not is_public_ip(parsed.hostname):
        raise HTTPException(status_code=400, detail="Image URL must point to a public host")

    request = Request(
        image_url,
        headers={
            "User-Agent": "cat-color-project/1.0",
            "Accept": "image/*",
        },
    )

    try:
        with urlopen(request, timeout=REMOTE_IMAGE_TIMEOUT) as response:
            content_type = response.headers.get("Content-Type", "")
            if content_type and not content_type.lower().startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail="The URL returned a webpage instead of an image. Please paste a direct JPG/PNG/WEBP image URL.",
                )

            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > REMOTE_IMAGE_MAX_BYTES:
                raise HTTPException(status_code=400, detail="Image URL is too large")

            data = response.read(REMOTE_IMAGE_MAX_BYTES + 1)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not download image from URL") from exc

    if len(data) > REMOTE_IMAGE_MAX_BYTES:
        raise HTTPException(status_code=400, detail="Image URL is too large")

    filename = Path(unquote(parsed.path)).name or parsed.hostname
    return read_image(data), filename


def build_model(checkpoint):
    names = checkpoint["class_names"]
    model_name = checkpoint.get("model_name", "efficientnet_b0")
    net = timm.create_model(model_name, pretrained=False, num_classes=len(names))
    net.load_state_dict(checkpoint["model_state_dict"])
    net.to(DEVICE)
    net.eval()
    return net


def build_transform(checkpoint):
    img_size = int(checkpoint.get("img_size", 300))
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


@app.on_event("startup")
def startup():
    global model, class_names, preprocess, confidence_threshold
    checkpoint = load_checkpoint()
    class_names = checkpoint["class_names"]
    preprocess = build_transform(checkpoint)
    model = build_model(checkpoint)
    confidence_threshold = load_confidence_threshold()


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML_PATH.read_text(encoding="utf-8")


def read_image(data: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc


def validate_top_k(top_k: int) -> int:
    if top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be >= 1")
    return min(top_k, len(class_names))


def predict_image(image: Image.Image, filename: str, top_k: int):
    if model is None or preprocess is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        scores, indices = torch.topk(probabilities, k=top_k)

    predictions = [
        {
            "class_index": int(index),
            "class_name": class_names[int(index)],
            "confidence": float(score),
        }
        for score, index in zip(scores.cpu(), indices.cpu())
    ]

    top_prediction = predictions[0]
    is_uncertain = top_prediction["confidence"] < confidence_threshold
    prediction = {
        "class_index": top_prediction["class_index"],
        "class_name": top_prediction["class_name"],
        "display_name": top_prediction["class_name"],
        "confidence": top_prediction["confidence"],
        "is_uncertain": False,
        "raw_class_index": top_prediction["class_index"],
        "raw_class_name": top_prediction["class_name"],
    }
    if is_uncertain:
        prediction.update(
            {
                "class_index": -1,
                "class_name": UNCERTAIN_CLASS_NAME,
                "display_name": UNCERTAIN_DISPLAY_NAME,
                "is_uncertain": True,
            }
        )

    return {
        "filename": filename,
        "prediction": prediction,
        "top_k": predictions,
        "threshold": confidence_threshold,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_path": str(MODEL_PATH),
        "classes": len(class_names),
        "class_names": class_names,
        "confidence_threshold": confidence_threshold,
    }


@app.get("/settings")
def get_settings():
    return {
        "confidence_threshold": confidence_threshold,
        "settings_path": str(SETTINGS_PATH),
    }


@app.put("/settings")
def update_settings(payload: ThresholdUpdate):
    global confidence_threshold

    try:
        confidence_threshold = normalize_threshold(payload.confidence_threshold)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    save_confidence_threshold(confidence_threshold)

    return {
        "status": "ok",
        "confidence_threshold": confidence_threshold,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3):
    top_k = validate_top_k(top_k)
    image = read_image(await file.read())
    return predict_image(image, file.filename, top_k)


@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...), top_k: int = 3):
    top_k = validate_top_k(top_k)
    if not files:
        raise HTTPException(status_code=400, detail="No image files uploaded")
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Upload up to 50 images at a time")

    results = []
    for file in files:
        image = read_image(await file.read())
        results.append(predict_image(image, file.filename, top_k))

    return {
        "count": len(results),
        "results": results,
    }


@app.post("/predict-url")
def predict_url(payload: ImageUrlRequest, top_k: int = 3):
    top_k = validate_top_k(top_k)
    image, filename = fetch_remote_image(payload.image_url)
    result = predict_image(image, filename, top_k)
    result["source_url"] = payload.image_url
    return result


@app.post("/predict-url-batch")
def predict_url_batch(payload: ImageUrlBatchRequest, top_k: int = 3):
    top_k = validate_top_k(top_k)
    image_urls = [item.strip() for item in payload.image_urls if item and item.strip()]
    if not image_urls:
        raise HTTPException(status_code=400, detail="No image URLs provided")
    if len(image_urls) > 50:
        raise HTTPException(status_code=400, detail="Submit up to 50 image URLs at a time")

    results = []
    success_count = 0
    for image_url in image_urls:
        try:
            image, filename = fetch_remote_image(image_url)
            result = predict_image(image, filename, top_k)
            result["source_url"] = image_url
            result["error"] = None
            results.append(result)
            success_count += 1
        except HTTPException as exc:
            parsed = urlparse(image_url)
            filename = Path(unquote(parsed.path)).name or parsed.hostname or image_url
            results.append(
                {
                    "filename": filename,
                    "source_url": image_url,
                    "error": exc.detail,
                }
            )

    return {
        "count": len(results),
        "success_count": success_count,
        "failure_count": len(results) - success_count,
        "results": results,
    }

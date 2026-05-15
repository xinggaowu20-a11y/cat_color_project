from io import BytesIO
from pathlib import Path

import timm
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from torchvision import transforms


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_efficientnet_b0_cat_color.pth"
INDEX_HTML_PATH = BASE_DIR / "index.html"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Cat Color Recognition API")

model = None
class_names = []
preprocess = None


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
    global model, class_names, preprocess
    checkpoint = load_checkpoint()
    class_names = checkpoint["class_names"]
    preprocess = build_transform(checkpoint)
    model = build_model(checkpoint)


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

    return {
        "filename": filename,
        "prediction": predictions[0],
        "top_k": predictions,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "classes": len(class_names),
        "class_names": class_names,
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

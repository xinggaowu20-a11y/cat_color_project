---
title: Cat Color Recognition
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Cat Color Recognition

EfficientNet-B0 cat color recognition app with a FastAPI backend and a browser UI.

## Classes

- Black_White_cat
- Black_cat
- Blue_White_cat
- Blue_cat
- Calico_cat
- Colorpoint_cat
- Orange_Cat
- Shaded_Golden
- Shaded_Silver
- Tortoiseshell
- White_cat

## Local Run

```powershell
powershell -ExecutionPolicy Bypass -File D:\cat_color_project\start_local_server.ps1
```

Then open:

```text
http://127.0.0.1:8000/
```

## API

Single image:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict?top_k=3" -F "file=@D:\path\to\cat.jpg"
```

Batch images:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict-batch?top_k=3" `
  -F "files=@D:\path\to\cat1.jpg" `
  -F "files=@D:\path\to\cat2.jpg"
```

## Docker

```powershell
docker build -t cat-color-api .
docker run --rm -p 7860:7860 cat-color-api
```

Open:

```text
http://127.0.0.1:7860/
```

## Hugging Face Spaces

Create a new Space with:

- SDK: Docker
- Visibility: Public or Private

Upload or push this repository to the Space. Hugging Face will build the Docker image and expose the app on port `7860`.

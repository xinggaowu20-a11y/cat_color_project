# Cat Color Recognition

基于 EfficientNet-B0 的猫咪颜色识别模型，当前模型支持 11 个类别：

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

## 本地运行

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

打开：

- Web 界面：http://127.0.0.1:8000/
- API 状态：http://127.0.0.1:8000/health
- 在线测试文档：http://127.0.0.1:8000/docs

## 接口调用

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict?top_k=3" `
  -F "file=@D:\path\to\cat.jpg"
```

返回示例：

```json
{
  "filename": "cat.jpg",
  "prediction": {
    "class_index": 1,
    "class_name": "Black_cat",
    "confidence": 0.98
  },
  "top_k": []
}
```

## 命令行预测

```powershell
python predict.py D:\path\to\cat.jpg --top-k 3
```

## Docker 部署

```powershell
docker build -t cat-color-api .
docker run --rm -p 8000:8000 cat-color-api
```

## 上传到 GitHub

```powershell
git init
git add .
git commit -m "Deploy cat color recognition API"
git branch -M main
git remote add origin https://github.com/xinggaowu20-a11y/cat_color_project.git
git push -u origin main
```

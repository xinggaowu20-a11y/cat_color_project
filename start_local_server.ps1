$ProjectDir = "D:\cat_color_project"
$PythonExe = "C:\Users\27561\.conda\envs\testenv\python.exe"

Set-Location $ProjectDir
& $PythonExe -m uvicorn app:app --host 127.0.0.1 --port 8000

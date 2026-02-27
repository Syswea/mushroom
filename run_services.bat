@echo off
TITLE 蘑菇毒性预测系统启动器
SETLOCAL

echo ==========================================
echo   蘑菇毒性预测系统 (Mushroom Prediction)
echo ==========================================

REM 检查 Conda 是否可用
where conda >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [错误] 未找到 Conda 命令，请确保 Conda 已安装并加入环境变量。
    pause
    exit /b
)

echo [信息] 正在激活 Conda 环境: mushroom...
call conda activate mushroom
if %ERRORLEVEL% neq 0 (
    echo [错误] 无法激活 'mushroom' 环境。
    echo 请先运行 'conda env create -f environment.yml' 创建环境。
    pause
    exit /b
)

echo [信息] 正在启动各服务组件...

REM 启动分类器 API (端口 8000)
echo - 正在新窗口启动分类器 API...
start "Mushroom - Classifier API" cmd /k "python src\classifier_api.py"

REM 启动 VLM API (端口 8001)
echo - 正在新窗口启动 VLM API...
start "Mushroom - VLM API" cmd /k "python src\imgprocess_api.py"

REM 启动 Streamlit 前端 (端口 8501)
echo - 正在新窗口启动 Streamlit 前端...
start "Mushroom - Frontend" cmd /k "streamlit run src\front.py"

echo.
echo [成功] 所有服务已尝试启动！
echo ------------------------------------------
echo 分类器 API 地址: http://127.0.0.1:8000
echo VLM API 地址:    http://127.0.0.1:8001
echo 前端界面地址:    http://127.0.0.1:8501
echo ------------------------------------------
echo 请在弹出的各窗口中查看详细运行日志。
echo 按任意键退出本启动器（不会关闭已启动的服务窗口）。
pause >nul

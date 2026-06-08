@echo off
echo ========================================
echo  智能动态定价与促销决策平台
echo  Smart Dynamic Pricing Platform
echo ========================================
echo.

cd /d "%~dp0"

echo [1/4] 检查 Python 环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到 Python，请先安装 Python 3.9+
    pause
    exit /b 1
)
python --version
echo.

echo [2/4] 检查虚拟环境...
if not exist "venv" (
    echo 正在创建虚拟环境...
    python -m venv venv
)
echo 虚拟环境已就绪
echo.

echo [3/4] 激活虚拟环境并安装依赖...
call venv\Scripts\activate.bat
pip install -r requirements.txt
echo.

echo [4/4] 启动服务...
echo.
echo 服务启动后请访问:
echo   API 文档: http://localhost:8000/docs
echo   健康检查: http://localhost:8000/health
echo.
echo 按 Ctrl+C 停止服务
echo.

cd smart_pricing
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

pause

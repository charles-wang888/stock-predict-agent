@echo off
chcp 65001 >nul
title 股票交易智能体系统
color 0A

echo ========================================
echo   股票交易智能体系统 - 启动脚本
echo ========================================
echo.

REM 切换到项目根目录（脚本所在目录的上一级）
cd /d "%~dp0\.."

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.8+
    echo.
    echo 下载地址: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo [1/3] 检查Python环境...
python --version
if errorlevel 1 (
    echo [错误] Python环境检查失败
    pause
    exit /b 1
)

echo.
echo [2/3] 检查依赖包...
python -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo [提示] 正在安装依赖包，请稍候...
    echo [提示] 这可能需要几分钟时间，请耐心等待...
    echo.
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 (
        echo.
        echo [错误] 依赖包安装失败
        echo [提示] 请检查网络连接，或手动运行: pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
    echo [成功] 依赖包安装完成
) else (
    echo [成功] 依赖包已安装
)

echo.
echo [3/3] 启动Web服务...
echo.
echo ========================================
echo   服务启动中，请稍候...
echo   启动成功后，请在浏览器访问:
echo   http://localhost:5000
echo.
echo   按 Ctrl+C 停止服务
echo ========================================
echo.

REM 启动应用
python main.py
if errorlevel 1 (
    echo.
    echo [错误] 服务启动失败
    echo [提示] 请检查错误信息，或查看README.md获取帮助
    echo.
    pause
    exit /b 1
)

pause


#!/bin/bash

echo "========================================"
echo "  股票交易智能体系统 - 启动脚本"
echo "========================================"
echo ""

# 切换到项目根目录（脚本所在目录的上一级）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "[错误] 未检测到Python，请先安装Python 3.8+"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

echo "[1/3] 检查Python环境..."
$PYTHON_CMD --version

echo ""
echo "[2/3] 检查依赖包..."
$PYTHON_CMD -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[提示] 正在安装依赖包，请稍候..."
    $PYTHON_CMD -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if [ $? -ne 0 ]; then
        echo "[错误] 依赖包安装失败"
        exit 1
    fi
else
    echo "[提示] 依赖包已安装"
fi

echo ""
echo "[3/3] 启动Web服务..."
echo ""
echo "========================================"
echo "  服务启动中，请稍候..."
echo "  启动成功后，请在浏览器访问:"
echo "  http://localhost:5000"
echo "========================================"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

$PYTHON_CMD main.py


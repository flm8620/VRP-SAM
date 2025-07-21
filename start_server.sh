#!/bin/bash
# VRP-SAM 推理服务快速启动脚本

echo "========================================="
echo "VRP-SAM 推理服务启动脚本"
echo "========================================="

# 默认参数
MODEL_PATH="/home/leman.feng/VRP-SAM-eval/logs/trn1_coco_mask_fold0.log/best_model.pt"
HOST="0.0.0.0"
PORT="58080"
DEVICE="cuda"
SAM_VERSION="vit_h"
BACKBONE="resnet50"

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "警告: 模型文件不存在: $MODEL_PATH"
    echo "请检查模型路径或训练模型"
    echo ""
fi

# 检查Python依赖
echo "检查Python依赖..."
python3 -c "import torch, flask, PIL, numpy, matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "错误: 缺少必要的Python依赖"
    echo "请安装: pip install torch flask pillow numpy matplotlib opencv-python requests"
    exit 1
fi

echo "依赖检查通过"
echo ""

# 显示启动信息
echo "启动参数:"
echo "  模型路径: $MODEL_PATH"
echo "  服务地址: http://$HOST:$PORT"
echo "  设备: $DEVICE"
echo "  SAM版本: $SAM_VERSION"
echo "  骨干网络: $BACKBONE"
echo ""

echo "可用的API端点:"
echo "  健康检查: http://$HOST:$PORT/health"
echo "  推理接口: http://$HOST:$PORT/predict (POST)"
echo "  服务信息: http://$HOST:$PORT/info"
echo ""

echo "启动服务器..."
echo "按 Ctrl+C 停止服务"
echo ""

# 启动服务器
python inference_server.py \
    --model_path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --device "$DEVICE" \
    --sam_version "$SAM_VERSION" \
    --backbone "$BACKBONE"

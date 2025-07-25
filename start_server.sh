#!/bin/bash

# VRP-SAM 推理服务器启动脚本 V2
# 支持预配置类别的推理服务

# 服务器配置
SERVER_HOST="0.0.0.0"
SERVER_PORT="58080"
MODEL_PATH="./checkpoint_vrpsam_resnet101.pt"
CONFIG_PATH="./support_classes_config-lekiwi3.json"
SAM_VERSION="vit_h"
BACKBONE="resnet101"
DEVICE="cuda"

echo "========================================"
echo "VRP-SAM 推理服务器"
echo "========================================"

# 检查配置文件
if [ ! -f "$CONFIG_PATH" ]; then
    echo "错误: 配置文件不存在: $CONFIG_PATH"
    echo "请确保配置文件存在并包含正确的类别定义"
    exit 1
fi

echo "配置信息:"
echo "  服务器地址: ${SERVER_HOST}:${SERVER_PORT}"
echo "  模型文件: $MODEL_PATH"
echo "  配置文件: $CONFIG_PATH"
echo "  SAM版本: $SAM_VERSION"
echo "  骨干网络: $BACKBONE"
echo "  设备: $DEVICE"
echo ""

# 检查模型文件
if [ ! -f "$MODEL_PATH" ]; then
    echo "警告: 模型文件不存在: $MODEL_PATH"
    exit 1
fi

echo "启动推理服务器..."
echo "按 Ctrl+C 停止服务器"
echo ""

# 启动服务器
python inference_server.py \
    --model_path "$MODEL_PATH" \
    --config_path "$CONFIG_PATH" \
    --sam_version "$SAM_VERSION" \
    --backbone "$BACKBONE" \
    --device "$DEVICE" \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT"

echo ""
echo "服务器已停止"

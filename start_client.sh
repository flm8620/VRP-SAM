#!/bin/bash

# VRP-SAM 客户端测试脚本
# 测试基于类别的推理功能

echo "======================================="
echo "VRP-SAM 客户端测试脚本"
echo "======================================="

# 默认参数
SERVER_URL="http://localhost:58080"
DATA_DIR="datasets/robot_data/test_data"
OUTPUT_DIR="./client_results"

echo "测试参数:"
echo "  服务器地址: $SERVER_URL"
echo "  数据目录: $DATA_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 检查数据目录是否存在
if [ ! -d "$DATA_DIR" ]; then
    echo "错误: 数据目录不存在: $DATA_DIR"
    echo "请检查数据路径"
    exit 1
fi

# 检查服务器是否运行
echo "检查服务器状态..."
response=$(curl -s -o /dev/null -w "%{http_code}" "$SERVER_URL/health")
if [ "$response" != "200" ]; then
    echo "错误: 无法连接到服务器 $SERVER_URL"
    echo "请先启动推理服务器: ./start_server_v2.sh"
    exit 1
fi

echo "✓ 服务器连接正常"
echo ""

# 首先列出可用类别
echo "获取可用类别..."
python inference_client.py \
    --server_url "$SERVER_URL" \
    --class_name dummy \
    --query_images dummy.jpg \
    --list_classes

echo ""

# 测试cube类别 - 单张图片
echo "=== 测试 1: table 类别 - 单张图片 ==="
python inference_client.py \
    --server_url "$SERVER_URL" \
    --class_name "table" \
    --query_images "$DATA_DIR/far_test/front_far_t5.00s_f000150.png" \
    --output_dir "$OUTPUT_DIR" \
    --use_all_support

echo ""

# 测试cube类别 - 多张图片
echo "=== 测试 2: cube 类别 - 批量图片 ==="
python inference_client.py \
    --server_url "$SERVER_URL" \
    --class_name "cube" \
    --query_images "$DATA_DIR/near_test/front_near_t3.00s_f000090.png" "$DATA_DIR/near_test/left_near_t11.00s_f000330.png" \
    --output_dir "$OUTPUT_DIR" \
    --max_batch_size 5

echo ""

echo ""
echo "=== 所有测试完成! ==="
echo "结果保存在: $OUTPUT_DIR"
echo ""
echo "可以查看以下文件:"
echo "  - *_binary_mask.png: 二值分割结果"
echo "  - *_probability.png: 概率热图"
echo "  - *_overlay.png: 叠加结果"
echo "  - *_stats.json: 统计信息"
echo "  - *_summary.json: 批次总结"

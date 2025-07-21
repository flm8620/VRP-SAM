#!/bin/bash
# VRP-SAM 客户端测试脚本

echo "========================================="
echo "VRP-SAM 客户端测试脚本"
echo "========================================="

# 默认参数
SERVER_URL="http://localhost:58080"
DATA_DIR="datasets/my_test/car_cube_cat"
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
curl -s "$SERVER_URL/health" > /dev/null
if [ $? -ne 0 ]; then
    echo "错误: 无法连接到服务器 $SERVER_URL"
    echo "请先启动推理服务器: ./start_server.sh"
    exit 1
fi

echo "服务器连接正常"
echo ""

# 测试cube类别
echo "测试 cube 类别..."
python inference_client.py \
    --server_url "$SERVER_URL" \
    --data_dir "$DATA_DIR" \
    --object_class "cube" \
    --output_dir "$OUTPUT_DIR" \
    --use_all_support

echo ""

# 测试car类别
echo "测试 car 类别..."
python inference_client.py \
    --server_url "$SERVER_URL" \
    --data_dir "$DATA_DIR" \
    --object_class "car" \
    --output_dir "$OUTPUT_DIR" \
    --use_all_support

echo ""
echo "客户端测试完成!"
echo "结果保存在: $OUTPUT_DIR"

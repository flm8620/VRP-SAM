#!/bin/bash

# VRP-SAM 数据集推理脚本
# 适用于 car_cube_cat 数据集结构

CUDA_VISIBLE_DEVICES=0
# 基本用法：处理所有类别，使用所有可用的支持图像
echo "=== 基本推理：处理 cube 和 table 两个类别 ==="
python simple_inference.py \
    --query_dir /home/gao/repos/VRP-SAM/datasets/robot_data/test_data/near_test \
    --support_dir /home/gao/repos/VRP-SAM/datasets/robot_data/cube_table_data \
    --object_classes cube table \
    --output_dir /home/gao/repos/VRP-SAM/datasets/robot_data/near_results/ \
    --model_path /home/gao/repos/VRP-SAM/checkpoint_vrpsam_resnet101.pt \
    --sam_version vit_h \
    --backbone resnet101 \
    --device cuda

echo ""
echo "=== 指定支持图像的推理示例 ==="
# 指定特定图像作为支持样本的示例
# python simple_inference.py \
#     --data_dir ./car_cube_cat \
#     --object_classes cube car \
#     --output_dir ./results_specific_support \
#     --support_images image1 image3 image5 \
#     --single_support \
#     --device cuda

echo ""
echo "推理完成！"

#!/bin/bash

# VRP-SAM 数据集推理脚本
# 适用于 car_cube_cat 数据集结构

CUDA_VISIBLE_DEVICES=4
# 基本用法：处理所有类别，使用所有可用的支持图像
echo "=== 基本推理：处理 cube 和 car 两个类别 ==="
python simple_inference.py \
    --data_dir /home/leman.feng/VRP-SAM-eval/datasets/my_test/car_cube_cat \
    --object_classes cube car \
    --output_dir /home/leman.feng/VRP-SAM-eval/datasets/my_test/results \
    --model_path /home/leman.feng/VRP-SAM-eval/datasets/best_model2.pt \
    --sam_version vit_h \
    --backbone resnet50 \
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
echo "推理完成！查看结果："
echo "- cube 结果: ./results/cube_results/"
echo "- car 结果:  ./results/car_results/"
echo "每个结果文件夹包含："
echo "  - support_info.txt: 支持图像信息"
echo "  - *_result.png: 综合分析图"
echo "  - *_prob_heatmap.png: 概率热图"
echo "  - *_prob.npy: 概率数组"
echo "  - *_mask.png: 二值mask"

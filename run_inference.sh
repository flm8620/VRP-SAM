#!/bin/bash

# VRP-SAM 简单推理示例脚本
CUDA_VISIBLE_DEVICES=4
# 运行推理
python simple_inference.py \
    --support_dir /home/leman.feng/VRP-SAM-eval/datasets/my_test/support_samples \
    --query_dir /home/leman.feng/VRP-SAM-eval/datasets/my_test/query_images \
    --output_dir /home/leman.feng/VRP-SAM-eval/datasets/my_test/inference_results2 \
    --model_path /home/leman.feng/VRP-SAM-eval/datasets/best_model2.pt \
    --sam_version vit_h \
    --backbone resnet50 \
    --device cuda

#!/bin/sh
# sleep 6h
PARTITION=Segmentation

GPU_ID=4,5,6,7

CUDA_VISIBLE_DEVICES=${GPU_ID} torchrun --nproc_per_node=4 --master_port=6224 train.py \
                        --epochs 50 \
                        --condition mask \
                        --lr 1e-4 \
                        --fold 0 \
                        --logpath trn1_coco_mask_fold0-resnet101 \
                        --backbone resnet101 \
                        --datapath /home/leman.feng/VRP-SAM-eval/datasets

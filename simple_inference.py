#!/usr/bin/env python3
"""
VRP-SAM Simple Inference Script
简化的推理脚本，用于在新图像上进行物体分割

输入：支持图像+mask，查询图像
输出：查询图像的分割mask，保存为PNG文件

使用方法:
python simple_inference.py --support_dir support_samples/ --query_dir query_images/ --output_dir results/
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# 导入模型
from model.VRP_encoder import VRP_encoder
from SAM2pred import SAM_pred
from common import utils


class VRPSAMInference:
    def __init__(self, model_path, sam_version='vit_h', backbone='resnet50', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 加载VRP模型
        args = self._create_args(backbone)
        self.vrp_model = VRP_encoder(args, backbone, False)
        
        if model_path and os.path.exists(model_path):
            print(f"Loading VRP model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            # 移除 'module.' 前缀（如果存在）
            checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            self.vrp_model.load_state_dict(checkpoint)
        else:
            print("Warning: No model checkpoint provided, using random weights")
            
        self.vrp_model.to(self.device)
        self.vrp_model.eval()
        
        # 加载SAM模型
        print(f"Loading SAM model: {sam_version}")
        self.sam_model = SAM_pred(sam_version)
        self.sam_model.to(self.device)
        self.sam_model.eval()
        
        # 图像预处理参数
        self.img_size = 512
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
    def _create_args(self, backbone):
        """创建模型需要的参数"""
        class Args:
            def __init__(self):
                self.condition = 'mask'
                self.backbone = backbone
                self.num_query = 50  # 添加缺失的 num_query 参数
                
        return Args()
    
    def load_image(self, image_path):
        """加载并预处理图像"""
        img = Image.open(image_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_array = np.array(img) / 255.0
        
        # 归一化
        for i in range(3):
            img_array[:, :, i] = (img_array[:, :, i] - self.mean[i]) / self.std[i]
        
        # 转为tensor
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        return img_tensor.unsqueeze(0)  # 添加batch维度
    
    def load_mask(self, mask_path):
        """加载mask"""
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        mask_array = np.array(mask)
        
        # 二值化
        mask_array = (mask_array > 128).astype(np.float32)
        
        # 转为tensor
        mask_tensor = torch.from_numpy(mask_array).float()
        return mask_tensor.unsqueeze(0)  # 添加batch维度
    
    def preprocess_support_data(self, support_dir, object_class, support_images=None):
        """预处理支持数据
        Args:
            support_dir: 包含图像和mask的根目录
            object_class: 物体类别 ('cube' 或 'car')
            support_images: 指定的支持图像列表，如果为None则使用所有可用图像
        """
        support_image_tensors = []
        support_mask_tensors = []
        used_support_files = []
        
        # 设置路径
        img_dir = Path(support_dir)
        mask_dir = img_dir / f"{object_class}_mask"
        
        if not img_dir.exists():
            raise ValueError(f"Image directory not found: {img_dir}")
        if not mask_dir.exists():
            raise ValueError(f"Mask directory not found: {mask_dir}")
        
        # 获取所有jpg图像文件
        all_img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.JPG"))
        
        if support_images is not None:
            # 用户指定了特定的支持图像
            selected_files = []
            for img_name in support_images:
                # 支持带扩展名和不带扩展名的输入
                if not img_name.endswith('.jpg') and not img_name.endswith('.JPG'):
                    img_file = img_dir / f"{img_name}.jpg"
                    if not img_file.exists():
                        img_file = img_dir / f"{img_name}.JPG"
                else:
                    img_file = img_dir / img_name
                
                if img_file.exists() and img_file in all_img_files:
                    selected_files.append(img_file)
                else:
                    print(f"Warning: Specified support image not found: {img_name}")
            
            if not selected_files:
                raise ValueError(f"None of the specified support images found: {support_images}")
            
            img_files = selected_files
        else:
            # 使用所有可用图像
            img_files = all_img_files
        
        # 处理每个图像文件
        for img_file in sorted(img_files):
            img_stem = img_file.stem
            
            # 查找对应的mask文件 (格式: xxx_mask.png)
            mask_file = mask_dir / f"{img_stem}_mask.png"
            if not mask_file.exists():
                mask_file = mask_dir / f"{img_stem}_mask.PNG"
            
            if mask_file.exists():
                print(f"Found {object_class} support pair: {img_file.name} -> {mask_file.name}")
                
                # 加载图像和mask
                img_tensor = self.load_image(str(img_file))
                mask_tensor = self.load_mask(str(mask_file))
                
                support_image_tensors.append(img_tensor)
                support_mask_tensors.append(mask_tensor)
                used_support_files.append(img_file.name)
            else:
                print(f"Warning: No {object_class} mask found for {img_file.name}")
        
        if not support_image_tensors:
            raise ValueError(f"No valid {object_class} support image-mask pairs found")
        
        # 合并所有支持数据
        support_imgs = torch.cat(support_image_tensors, dim=0)
        support_masks = torch.cat(support_mask_tensors, dim=0)
        
        return support_imgs, support_masks, used_support_files
    
    def predict_single_image(self, query_img_path, support_imgs, support_masks, use_all_support=True):
        """对单张查询图像进行预测"""
        # 加载查询图像
        query_img = self.load_image(query_img_path)
        query_name = [os.path.basename(query_img_path)]
        
        # 移动到GPU
        query_img = query_img.to(self.device)
        support_imgs = support_imgs.to(self.device)
        support_masks = support_masks.to(self.device)
        
        with torch.no_grad():
            if use_all_support and len(support_imgs) > 1:
                # 使用多个支持样本进行预测（类似于n-shot学习）
                print(f"Using {len(support_imgs)} support samples for prediction")
                protos_list = []
                
                # 对每个支持样本提取prototypes
                for i in range(len(support_imgs)):
                    support_img = support_imgs[i:i+1]
                    support_mask = support_masks[i:i+1]
                    
                    protos, _ = self.vrp_model(
                        'mask',  # condition 
                        query_img, 
                        support_img, 
                        support_mask, 
                        training=False
                    )
                    protos_list.append(protos)
                
                # 合并多个prototypes
                combined_protos = torch.cat(protos_list, dim=1)
                
            else:
                # 只使用第一个支持样本
                print("Using single support sample for prediction")
                support_img = support_imgs[0:1]
                support_mask = support_masks[0:1]
                
                # VRP编码器提取prototypes
                combined_protos, _ = self.vrp_model(
                    'mask',  # condition 
                    query_img, 
                    support_img, 
                    support_mask, 
                    training=False
                )
            
            # SAM预测
            low_masks, pred_mask = self.sam_model(query_img, query_name, combined_protos)
            
            # 获取概率分布
            prob_mask = torch.sigmoid(low_masks)
            
            # 使用sigmoid阈值得到二值mask
            binary_mask = prob_mask > 0.5
            binary_mask = binary_mask.float()
        
        return binary_mask.squeeze().cpu().numpy(), prob_mask.squeeze().cpu().numpy()
    
    def save_visualization(self, query_img_path, pred_mask, prob_mask, output_path, support_files, object_class):
        """保存可视化结果，包括概率热图和支持文件信息"""
        # 加载原始查询图像用于可视化
        original_img = Image.open(query_img_path).convert('RGB')
        original_img = original_img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_array = np.array(original_img)
        
        # 创建彩色mask
        mask_colored = np.zeros_like(img_array)
        mask_colored[:, :, 0] = pred_mask * 255  # 红色通道
        
        # 叠加
        alpha = 0.5
        overlay = img_array * (1 - alpha) + mask_colored * alpha
        overlay = overlay.astype(np.uint8)

        save_prob = True
        
        if save_prob:
            # 创建带概率热图的6格图像：原图 | 二值mask | 叠加 | 概率热图 | 概率叠加 | 概率分布
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # 添加总标题，包含支持文件信息
            support_info = f"Object: {object_class.upper()} | Support files: {', '.join(support_files)}"
            fig.suptitle(support_info, fontsize=14, y=0.98)
            
            # 第一行：基本结果
            axes[0, 0].imshow(img_array)
            axes[0, 0].set_title('Original Image', fontsize=12)
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(pred_mask, cmap='gray')
            axes[0, 1].set_title('Binary Mask (Threshold=0.5)', fontsize=12)
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(overlay)
            axes[0, 2].set_title('Binary Overlay', fontsize=12)
            axes[0, 2].axis('off')
            
            # 第二行：概率分析
            # 概率热图
            im1 = axes[1, 0].imshow(prob_mask, cmap='jet', vmin=0, vmax=1)
            axes[1, 0].set_title('Probability Heatmap', fontsize=12)
            axes[1, 0].axis('off')
            # 添加colorbar
            cbar1 = plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
            cbar1.set_label('Probability', rotation=270, labelpad=15)
            
            # 概率叠加在原图上
            # 创建透明度基于概率的叠加
            prob_colored = plt.cm.jet(prob_mask)[:, :, :3]  # 取RGB，忽略alpha
            prob_overlay = img_array * 0.6 + prob_colored * 255 * 0.4
            prob_overlay = np.clip(prob_overlay, 0, 255).astype(np.uint8)
            
            axes[1, 1].imshow(prob_overlay)
            axes[1, 1].set_title('Probability Overlay', fontsize=12)
            axes[1, 1].axis('off')
            
            # 概率分布直方图
            axes[1, 2].hist(prob_mask.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
            axes[1, 2].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold=0.5')
            axes[1, 2].set_xlabel('Probability', fontsize=10)
            axes[1, 2].set_ylabel('Pixel Count', fontsize=10)
            axes[1, 2].set_title('Probability Distribution', fontsize=12)
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_prob = prob_mask.mean()
            std_prob = prob_mask.std()
            high_conf = (prob_mask > 0.8).sum() / prob_mask.size * 100
            low_conf = (prob_mask < 0.2).sum() / prob_mask.size * 100
            uncertain = ((prob_mask > 0.3) & (prob_mask < 0.7)).sum() / prob_mask.size * 100
            
            stats_text = f'Mean: {mean_prob:.3f}\nStd: {std_prob:.3f}\nHigh Conf (>0.8): {high_conf:.1f}%\nLow Conf (<0.2): {low_conf:.1f}%\nUncertain (0.3-0.7): {uncertain:.1f}%'
            axes[1, 2].text(0.02, 0.98, stats_text, transform=axes[1, 2].transAxes, 
                           verticalalignment='top', fontsize=8, 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
        else:
            # 创建拼接图像：原图 | mask | 叠加
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(img_array)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(pred_mask, cmap='gray')
            axes[1].set_title('Predicted Mask')
            axes[1].axis('off')
            
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 保存单独的mask和概率文件
        mask_path = output_path.replace('.png', '_mask.png')
        mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
        mask_img.save(mask_path)
        
        if save_prob:
            # 保存概率数组
            prob_path = output_path.replace('.png', '_prob.npy')
            np.save(prob_path, prob_mask)
            
            # 保存单独的概率热图
            prob_heatmap_path = output_path.replace('.png', '_prob_heatmap.png')
            plt.figure(figsize=(10, 8))
            im = plt.imshow(prob_mask, cmap='jet', vmin=0, vmax=1)
            plt.title(f'{object_class.upper()} Segmentation Probability Heatmap', fontsize=14)
            plt.axis('off')
            
            # 添加详细的colorbar
            cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
            cbar.set_label('Segmentation Probability', rotation=270, labelpad=20, fontsize=12)
            cbar.ax.tick_params(labelsize=10)
            
            # 添加概率级别标注
            prob_levels = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
            prob_labels = ['Background\n(0.0)', 'Low\n(0.2)', 'Uncertain\n(0.4)', 
                          'Threshold\n(0.5)', 'Likely\n(0.6)', 'High\n(0.8)', 'Certain\n(1.0)']
            cbar.set_ticks(prob_levels)
            cbar.set_ticklabels(prob_labels)
            
            plt.savefig(prob_heatmap_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    def run_inference(self, support_dir, object_classes, output_dir, use_all_support=True, support_images=None):
        """运行完整推理流程
        Args:
            support_dir: 包含图像和mask的根目录
            object_classes: 要处理的物体类别列表，如['cube', 'car']
            output_dir: 输出根目录
            use_all_support: 是否使用所有支持样本
            support_images: 指定的支持图像列表
        """
        # 创建输出根目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有查询图像（所有jpg文件）
        support_path = Path(support_dir)
        query_files = list(support_path.glob("*.jpg")) + list(support_path.glob("*.JPG"))
        
        if not query_files:
            raise ValueError(f"No query images (jpg files) found in {support_dir}")
        
        print(f"Found {len(query_files)} total images in dataset")
        
        # 为每个物体类别进行推理
        for object_class in object_classes:
            print(f"\n{'='*50}")
            print(f"Processing object class: {object_class.upper()}")
            print(f"{'='*50}")
            
            # 创建该类别的输出目录
            class_output_dir = Path(output_dir) / f"{object_class}_results"
            os.makedirs(class_output_dir, exist_ok=True)
            
            try:
                # 预处理支持数据
                print(f"Loading {object_class} support data...")
                support_imgs, support_masks, used_support_files = self.preprocess_support_data(
                    support_dir, object_class, support_images
                )
                print(f"Loaded {len(support_imgs)} {object_class} support samples: {used_support_files}")
                
                # 创建info文件记录支持文件
                info_file = class_output_dir / "support_info.txt"
                with open(info_file, 'w') as f:
                    f.write(f"Object Class: {object_class}\n")
                    f.write(f"Support Images Used: {', '.join(used_support_files)}\n")
                    f.write(f"Total Support Samples: {len(support_imgs)}\n")
                    f.write(f"Use All Support: {use_all_support}\n")
                    f.write(f"Query Images: All jpg files in {support_dir}\n")
                
                # 确定查询图像（排除支持图像）
                query_images = []
                for img_file in query_files:
                    if img_file.name not in used_support_files:
                        query_images.append(img_file)
                
                print(f"Query images for {object_class}: {len(query_images)} files")
                
                # 对每张查询图像进行预测
                for i, query_file in enumerate(sorted(query_images)):
                    print(f"Processing {object_class} {i+1}/{len(query_images)}: {query_file.name}")
                    
                    try:
                        # 预测
                        pred_mask, prob_mask = self.predict_single_image(
                            str(query_file), 
                            support_imgs, 
                            support_masks,
                            use_all_support=use_all_support
                        )
                        
                        # 保存结果
                        output_path = class_output_dir / f"{query_file.stem}_{object_class}_result.png"
                        self.save_visualization(
                            str(query_file), pred_mask, prob_mask, str(output_path), 
                            used_support_files, object_class
                        )
                        
                        print(f"  Saved result to {output_path}")
                        print(f"  - Probability data: {query_file.stem}_{object_class}_prob.npy")
                        print(f"  - Probability heatmap: {query_file.stem}_{object_class}_prob_heatmap.png")
                        
                    except Exception as e:
                        print(f"  Error processing {query_file.name}: {e}")
                        continue
                
                print(f"Completed {object_class} inference! Results saved in {class_output_dir}")
                
            except Exception as e:
                print(f"Error processing {object_class}: {e}")
                continue
        
        print(f"\nAll inference completed! Results saved in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='VRP-SAM Simple Inference')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing images and mask subdirectories (e.g., car_cube_cat)')
    parser.add_argument('--object_classes', type=str, nargs='+', required=True,
                        help='Object classes to process (e.g., cube car)')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='Output directory for results')
    parser.add_argument('--model_path', type=str, 
                        default='/home/leman.feng/VRP-SAM-eval/logs/trn1_coco_mask_fold0.log/best_model.pt',
                        help='Path to trained VRP model checkpoint')
    parser.add_argument('--sam_version', type=str, default='vit_h',
                        choices=['vit_h', 'vit_l'], help='SAM model version')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['vgg16', 'resnet50', 'resnet101'], help='VRP backbone')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for inference')
    parser.add_argument('--use_all_support', action='store_true',
                        help='Use all support samples (n-shot learning) instead of just the first one')
    parser.add_argument('--single_support', action='store_true',
                        help='Use only the first support sample (1-shot learning)')
    parser.add_argument('--support_images', type=str, nargs='*', default=None,
                        help='Specific support images to use (without extension), if not provided, use all available')
    
    args = parser.parse_args()
    
    # 设置随机种子
    utils.fix_randseed(0)
    
    # 确定是否使用所有支持样本
    if args.single_support:
        use_all_support = False
        print("Using single support sample mode (1-shot learning)")
    else:
        use_all_support = True
        print("Using all available support samples mode (n-shot learning)")
    
    if args.support_images:
        print(f"Using specified support images: {args.support_images}")
    else:
        print("Using all available support images")
    
    # 创建推理器
    inferencer = VRPSAMInference(
        model_path=args.model_path,
        sam_version=args.sam_version,
        backbone=args.backbone,
        device=args.device
    )
    
    # 运行推理
    inferencer.run_inference(
        support_dir=args.data_dir,
        object_classes=args.object_classes,
        output_dir=args.output_dir,
        use_all_support=use_all_support,
        support_images=args.support_images
    )


if __name__ == '__main__':
    main()

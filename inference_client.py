#!/usr/bin/env python3
"""
VRP-SAM 客户端示例
演示如何与VRP-SAM推理服务器进行通信

功能：
1. 向推理服务器发送请求
2. 接收并显示分割结果
3. 保存结果图像

使用方法:
python inference_client.py --server_url http://localhost:8080 --data_dir datasets/my_test/car_cube_cat --object_class cube
"""

import os
import json
import base64
import io
import argparse
import time
from pathlib import Path
import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class VRPSAMClient:
    def __init__(self, server_url):
        self.server_url = server_url.rstrip('/')
        self.session = requests.Session()
        
    def image_to_base64(self, image_path):
        """将图像文件转换为base64字符串"""
        with open(image_path, 'rb') as f:
            image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            return image_base64
    
    def base64_to_image(self, base64_str):
        """将base64字符串转换为PIL图像"""
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        return image
    
    def check_server_health(self):
        """检查服务器健康状态"""
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"Server is healthy! Device: {data.get('device', 'unknown')}")
                return True
            else:
                print(f"Server health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"Cannot connect to server: {e}")
            return False
    
    def get_server_info(self):
        """获取服务器信息"""
        try:
            response = self.session.get(f"{self.server_url}/info", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print("Server Information:")
                for key, value in data.items():
                    print(f"  {key}: {value}")
                return data
            else:
                print(f"Failed to get server info: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error getting server info: {e}")
            return None
    
    def predict(self, query_image_path, support_image_paths, support_mask_paths, use_all_support=True):
        """发送推理请求"""
        try:
            # 准备请求数据
            print(f"Preparing request...")
            print(f"Query image: {query_image_path}")
            print(f"Support images: {support_image_paths}")
            print(f"Support masks: {support_mask_paths}")
            
            # 编码图像
            query_base64 = self.image_to_base64(query_image_path)
            
            support_images_base64 = []
            for img_path in support_image_paths:
                support_images_base64.append(self.image_to_base64(img_path))
            
            support_masks_base64 = []
            for mask_path in support_mask_paths:
                support_masks_base64.append(self.image_to_base64(mask_path))
            
            # 构建请求
            request_data = {
                'query_image': query_base64,
                'support_images': support_images_base64,
                'support_masks': support_masks_base64,
                'use_all_support': use_all_support
            }
            
            print(f"Sending request to server...")
            start_time = time.time()
            
            # 发送请求
            response = self.session.post(
                f"{self.server_url}/predict",
                json=request_data,
                timeout=60  # 推理可能需要较长时间
            )
            
            request_time = time.time() - start_time
            print(f"Request completed in {request_time:.3f}s")
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success', False):
                    print(f"Inference successful!")
                    print(f"Server inference time: {result.get('inference_time', 0):.3f}s")
                    print(f"Support samples used: {result.get('support_samples_used', 0)}")
                    
                    # 打印统计信息
                    stats = result.get('statistics', {})
                    print(f"Statistics:")
                    print(f"  Mean probability: {stats.get('mean_probability', 0):.3f}")
                    print(f"  Std probability: {stats.get('std_probability', 0):.3f}")
                    print(f"  High confidence (>0.8): {stats.get('high_confidence_ratio', 0)*100:.1f}%")
                    print(f"  Low confidence (<0.2): {stats.get('low_confidence_ratio', 0)*100:.1f}%")
                    print(f"  Positive pixels: {stats.get('positive_pixels', 0)}/{stats.get('total_pixels', 0)}")
                    
                    return result
                else:
                    print(f"Inference failed: {result.get('error', 'Unknown error')}")
                    return None
            else:
                print(f"Server error: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"Error details: {error_data.get('error', 'Unknown error')}")
                except:
                    print(f"Response text: {response.text}")
                return None
                
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None
    
    def save_results(self, result, output_dir, query_name):
        """保存推理结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        results_data = result.get('results', {})
        
        # 保存二值mask
        if 'binary_mask' in results_data:
            binary_mask = self.base64_to_image(results_data['binary_mask'])
            binary_path = os.path.join(output_dir, f"{query_name}_binary_mask.png")
            binary_mask.save(binary_path)
            print(f"Saved binary mask: {binary_path}")
        
        # 保存概率热图
        if 'probability_heatmap' in results_data:
            prob_heatmap = self.base64_to_image(results_data['probability_heatmap'])
            prob_path = os.path.join(output_dir, f"{query_name}_probability.png")
            prob_heatmap.save(prob_path)
            print(f"Saved probability heatmap: {prob_path}")
        
        # 保存叠加图像
        if 'overlay' in results_data:
            overlay = self.base64_to_image(results_data['overlay'])
            overlay_path = os.path.join(output_dir, f"{query_name}_overlay.png")
            overlay.save(overlay_path)
            print(f"Saved overlay: {overlay_path}")
        
        # 保存统计信息
        stats_path = os.path.join(output_dir, f"{query_name}_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Saved statistics: {stats_path}")


def find_support_data(data_dir, object_class):
    """查找支持数据（图像和mask）"""
    data_path = Path(data_dir)
    mask_dir = data_path / f"{object_class}_mask"
    
    support_images = []
    support_masks = []
    
    # 查找所有jpg图像
    img_files = list(data_path.glob("*.jpg")) + list(data_path.glob("*.JPG"))
    
    for img_file in sorted(img_files):
        img_stem = img_file.stem
        mask_file = mask_dir / f"{img_stem}_mask.png"
        
        if mask_file.exists():
            support_images.append(str(img_file))
            support_masks.append(str(mask_file))
    
    return support_images, support_masks


def main():
    parser = argparse.ArgumentParser(description='VRP-SAM Client Example')
    parser.add_argument('--server_url', type=str, default='http://localhost:8080',
                        help='URL of the inference server')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing test data')
    parser.add_argument('--object_class', type=str, required=True,
                        help='Object class to test (e.g., cube, car)')
    parser.add_argument('--query_image', type=str, default=None,
                        help='Specific query image to test (if not provided, use all non-support images)')
    parser.add_argument('--output_dir', type=str, default='./client_results',
                        help='Output directory for results')
    parser.add_argument('--use_all_support', action='store_true',
                        help='Use all support samples (n-shot learning)')
    parser.add_argument('--single_support', action='store_true',
                        help='Use only the first support sample (1-shot learning)')
    parser.add_argument('--max_support', type=int, default=None,
                        help='Maximum number of support samples to use')
    
    args = parser.parse_args()
    
    # 创建客户端
    client = VRPSAMClient(args.server_url)
    
    # 检查服务器状态
    print("Checking server health...")
    if not client.check_server_health():
        print("Server is not accessible. Please start the server first.")
        return
    
    # 获取服务器信息
    print("\nGetting server information...")
    client.get_server_info()
    
    # 查找支持数据
    print(f"\nLooking for {args.object_class} support data in {args.data_dir}...")
    support_images, support_masks = find_support_data(args.data_dir, args.object_class)
    
    if not support_images:
        print(f"No {args.object_class} support data found!")
        return
    
    print(f"Found {len(support_images)} support samples:")
    for img, mask in zip(support_images, support_masks):
        print(f"  {os.path.basename(img)} -> {os.path.basename(mask)}")
    
    # 限制支持样本数量
    if args.max_support and len(support_images) > args.max_support:
        support_images = support_images[:args.max_support]
        support_masks = support_masks[:args.max_support]
        print(f"Limited to {args.max_support} support samples")
    
    # 确定使用模式
    if args.single_support:
        use_all_support = False
        print("Using single support mode (1-shot learning)")
    else:
        use_all_support = True
        print("Using all support samples mode (n-shot learning)")
    
    # 确定查询图像
    data_path = Path(args.data_dir)
    if args.query_image:
        # 使用指定的查询图像
        query_path = data_path / args.query_image
        if not query_path.exists():
            print(f"Query image not found: {query_path}")
            return
        query_images = [str(query_path)]
    else:
        # 使用所有非支持图像作为查询图像
        all_images = list(data_path.glob("*.jpg")) + list(data_path.glob("*.JPG"))
        support_basenames = [os.path.basename(img) for img in support_images]
        query_images = [str(img) for img in all_images if os.path.basename(img) not in support_basenames]
    
    if not query_images:
        print("No query images found!")
        return
    
    print(f"\nFound {len(query_images)} query images")
    
    # 创建输出目录
    output_dir = Path(args.output_dir) / f"{args.object_class}_client_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 对每个查询图像进行推理
    success_count = 0
    total_time = 0
    
    for i, query_img_path in enumerate(query_images):
        query_name = Path(query_img_path).stem
        print(f"\n{'='*60}")
        print(f"Processing {i+1}/{len(query_images)}: {os.path.basename(query_img_path)}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # 发送推理请求
        result = client.predict(
            query_img_path,
            support_images,
            support_masks,
            use_all_support=use_all_support
        )
        
        if result:
            # 保存结果
            client.save_results(result, str(output_dir), query_name)
            success_count += 1
            
            inference_time = time.time() - start_time
            total_time += inference_time
            print(f"Total request time: {inference_time:.3f}s")
        else:
            print(f"Failed to process {query_img_path}")
    
    # 打印总结
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {success_count}/{len(query_images)}")
    print(f"Success rate: {success_count/len(query_images)*100:.1f}%")
    print(f"Average time per image: {total_time/len(query_images):.3f}s")
    print(f"Results saved in: {output_dir}")
    
    if success_count > 0:
        print(f"\nClient testing completed successfully!")
    else:
        print(f"\nAll requests failed!")


if __name__ == '__main__':
    main()

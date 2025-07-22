#!/usr/bin/env python3
"""
VRP-SAM 新客户端
使用预配置类别进行简化推理

功能：
1. 向推理服务器发送基于类别的请求
2. 支持批量查询图像
3. 保存结果图像

使用方法:
python inference_client_v2.py --server_url http://localhost:8080 --class_name cube --query_images img1.jpg img2.jpg
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
import glob


class VRPSAMClientV2:
    def __init__(self, server_url):
        self.server_url = server_url.rstrip('/')
        self.session = requests.Session()
        
    def image_to_base64(self, image_path):
        """将图像文件转换为base64字符串"""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
    
    def base64_to_image(self, base64_str):
        """将base64字符串转换为PIL图像"""
        try:
            image_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_data))
            return image
        except Exception as e:
            print(f"Error decoding base64 image: {e}")
            return None
    
    def check_server_health(self):
        """检查服务器健康状态"""
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Server is healthy! Device: {data.get('device', 'unknown')}")
                return True
            else:
                print(f"✗ Server health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Cannot connect to server: {e}")
            return False
    
    def get_available_classes(self):
        """获取可用的识别类别"""
        try:
            response = self.session.get(f"{self.server_url}/classes", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('success', False):
                    return data.get('classes', {})
                else:
                    print(f"Failed to get classes: {data.get('error', 'Unknown error')}")
                    return {}
            else:
                print(f"Server error getting classes: {response.status_code}")
                return {}
        except Exception as e:
            print(f"Error getting classes: {e}")
            return {}
    
    def get_server_info(self):
        """获取服务器信息"""
        try:
            response = self.session.get(f"{self.server_url}/info", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print("=== Server Information ===")
                for key, value in data.items():
                    if key == 'available_classes':
                        print(f"  {key}: {', '.join(value) if value else 'None'}")
                    else:
                        print(f"  {key}: {value}")
                return data
            else:
                print(f"Failed to get server info: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error getting server info: {e}")
            return None
    
    def predict_with_class(self, query_image_paths, class_name, use_all_support=None):
        """使用预配置类别进行推理"""
        try:
            print(f"Preparing request for class '{class_name}'...")
            print(f"Query images: {len(query_image_paths)} files")
            
            # 编码查询图像
            query_images_base64 = []
            for img_path in query_image_paths:
                img_base64 = self.image_to_base64(img_path)
                if img_base64 is None:
                    print(f"Failed to encode {img_path}, skipping")
                    continue
                query_images_base64.append(img_base64)
            
            if not query_images_base64:
                print("No valid query images to process")
                return None
            
            # 构建请求
            request_data = {
                'class_name': class_name,
                'query_images': query_images_base64
            }
            
            if use_all_support is not None:
                request_data['use_all_support'] = use_all_support
            
            print(f"Sending request to server...")
            start_time = time.time()
            
            # 发送请求
            response = self.session.post(
                f"{self.server_url}/predict",
                json=request_data,
                timeout=120  # 批量推理可能需要更长时间
            )
            
            request_time = time.time() - start_time
            print(f"Request completed in {request_time:.3f}s")
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success', False):
                    print(f"✓ Inference successful!")
                    print(f"  Server inference time: {result.get('inference_time', 0):.3f}s")
                    print(f"  Class: {result.get('class_name', 'unknown')}")
                    print(f"  Query count: {result.get('query_count', 0)}")
                    print(f"  Support samples used: {result.get('support_samples_used', 0)}")
                    
                    # 打印每个结果的简要统计
                    results = result.get('results', [])
                    for i, res in enumerate(results):
                        stats = res.get('statistics', {})
                        pos_pixels = stats.get('positive_pixels', 0)
                        total_pixels = stats.get('total_pixels', 1)
                        coverage = pos_pixels / total_pixels * 100
                        mean_prob = stats.get('mean_probability', 0)
                        print(f"  Result {i+1}: {coverage:.1f}% coverage, {mean_prob:.3f} avg prob")
                    
                    return result
                else:
                    print(f"✗ Inference failed: {result.get('error', 'Unknown error')}")
                    return None
            else:
                print(f"✗ Server error: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"  Error details: {error_data.get('error', 'Unknown error')}")
                    if 'available_classes' in error_data:
                        print(f"  Available classes: {error_data['available_classes']}")
                except:
                    print(f"  Response text: {response.text}")
                return None
                
        except Exception as e:
            print(f"✗ Error during prediction: {e}")
            return None
    
    def save_results(self, result, output_dir, query_names):
        """保存推理结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        results = result.get('results', [])
        class_name = result.get('class_name', 'unknown')
        
        saved_files = []
        
        for i, (res, query_name) in enumerate(zip(results, query_names)):
            result_data = res.get('results', {})
            
            # 保存二值mask
            if 'binary_mask' in result_data:
                binary_mask = self.base64_to_image(result_data['binary_mask'])
                binary_path = os.path.join(output_dir, f"{query_name}_binary_mask.png")
                binary_mask.save(binary_path)
                saved_files.append(binary_path)
            
            # 保存概率热图
            if 'probability_heatmap' in result_data:
                prob_heatmap = self.base64_to_image(result_data['probability_heatmap'])
                prob_path = os.path.join(output_dir, f"{query_name}_probability.png")
                prob_heatmap.save(prob_path)
                saved_files.append(prob_path)
            
            # 保存叠加图像
            if 'overlay' in result_data:
                overlay = self.base64_to_image(result_data['overlay'])
                overlay_path = os.path.join(output_dir, f"{query_name}_overlay.png")
                overlay.save(overlay_path)
                saved_files.append(overlay_path)
            
            # 保存统计信息
            stats_path = os.path.join(output_dir, f"{query_name}_stats.json")
            with open(stats_path, 'w') as f:
                json.dump(res, f, indent=2)
            saved_files.append(stats_path)
        
        # 保存总体结果信息
        summary_path = os.path.join(output_dir, f"{class_name}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(result, f, indent=2)
        saved_files.append(summary_path)
        
        print(f"Results saved: {len(saved_files)} files in {output_dir}")
        return saved_files


def find_query_images(paths_or_patterns):
    """查找查询图像文件"""
    image_files = []
    
    for path_or_pattern in paths_or_patterns:
        if os.path.isfile(path_or_pattern):
            # 直接指定的文件
            image_files.append(path_or_pattern)
        elif os.path.isdir(path_or_pattern):
            # 目录，查找所有图像文件
            dir_path = Path(path_or_pattern)
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(glob.glob(str(dir_path / ext)))
        else:
            # 可能是通配符模式
            matches = glob.glob(path_or_pattern)
            image_files.extend(matches)
    
    # 去重并排序
    image_files = sorted(list(set(image_files)))
    return image_files


def main():
    parser = argparse.ArgumentParser(description='VRP-SAM Client V2 - Class-based Inference')
    parser.add_argument('--server_url', type=str, default='http://localhost:8080',
                        help='URL of the inference server')
    parser.add_argument('--class_name', type=str, required=True,
                        help='Object class name for inference (e.g., cube, car, table)')
    parser.add_argument('--query_images', nargs='+', required=True,
                        help='Query image files, directories, or patterns')
    parser.add_argument('--output_dir', type=str, default='./client_v2_results',
                        help='Output directory for results')
    parser.add_argument('--use_all_support', action='store_true',
                        help='Use all support samples (default: use server config)')
    parser.add_argument('--single_support', action='store_true',
                        help='Use only single support sample')
    parser.add_argument('--max_batch_size', type=int, default=10,
                        help='Maximum number of images to process in one batch')
    parser.add_argument('--list_classes', action='store_true',
                        help='List available classes and exit')
    
    args = parser.parse_args()
    
    # 创建客户端
    client = VRPSAMClientV2(args.server_url)
    
    # 检查服务器状态
    print("=== VRP-SAM Client V2 ===")
    print("Checking server health...")
    if not client.check_server_health():
        print("Server is not accessible. Please start the server first.")
        return
    
    # 获取服务器信息
    print("\nGetting server information...")
    client.get_server_info()
    
    # 获取可用类别
    print("\nGetting available classes...")
    available_classes = client.get_available_classes()
    
    if not available_classes:
        print("No classes available on server!")
        return
    
    print("Available classes:")
    for class_name, info in available_classes.items():
        print(f"  - {class_name}: {info.get('description', 'No description')} ({info.get('support_count', 0)} samples)")
    
    # 如果只是列出类别，则退出
    if args.list_classes:
        return
    
    # 检查指定类别是否存在
    if args.class_name not in available_classes:
        print(f"\nError: Class '{args.class_name}' not found!")
        print(f"Available classes: {list(available_classes.keys())}")
        return
    
    # 查找查询图像
    print(f"\nLooking for query images...")
    query_image_paths = find_query_images(args.query_images)
    
    if not query_image_paths:
        print("No query images found!")
        return
    
    print(f"Found {len(query_image_paths)} query images:")
    for i, img_path in enumerate(query_image_paths[:5]):  # 只显示前5个
        print(f"  {i+1}. {os.path.basename(img_path)}")
    if len(query_image_paths) > 5:
        print(f"  ... and {len(query_image_paths) - 5} more")
    
    # 确定推理模式
    use_all_support = None
    if args.use_all_support:
        use_all_support = True
    elif args.single_support:
        use_all_support = False
    
    # 创建输出目录
    output_dir = Path(args.output_dir) / f"{args.class_name}_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 批量处理
    total_images = len(query_image_paths)
    batch_size = args.max_batch_size
    total_batches = (total_images + batch_size - 1) // batch_size
    
    success_count = 0
    total_time = 0
    
    print(f"\n=== Starting Inference ===")
    print(f"Class: {args.class_name}")
    print(f"Total images: {total_images}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches: {total_batches}")
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_images)
        batch_paths = query_image_paths[start_idx:end_idx]
        
        print(f"\n--- Batch {batch_idx + 1}/{total_batches} ---")
        print(f"Processing images {start_idx + 1}-{end_idx} of {total_images}")
        
        start_time = time.time()
        
        # 发送推理请求
        result = client.predict_with_class(
            batch_paths, 
            args.class_name, 
            use_all_support=use_all_support
        )
        
        if result:
            # 生成查询名称
            query_names = [f"batch{batch_idx + 1}_{Path(p).stem}" for p in batch_paths]
            
            # 保存结果
            saved_files = client.save_results(result, str(output_dir), query_names)
            success_count += len(batch_paths)
            
            batch_time = time.time() - start_time
            total_time += batch_time
            
            print(f"✓ Batch completed in {batch_time:.3f}s")
            print(f"  Files saved: {len(saved_files)}")
        else:
            print(f"✗ Batch failed")
    
    # 打印总结
    print(f"\n=== Summary ===")
    print(f"Total images processed: {success_count}/{total_images}")
    if total_images > 0:
        print(f"Success rate: {success_count/total_images*100:.1f}%")
        print(f"Average time per image: {total_time/total_images:.3f}s")
    print(f"Results saved in: {output_dir}")
    
    if success_count > 0:
        print(f"\n✓ Processing completed successfully!")
        print(f"Check the results in: {output_dir}")
    else:
        print(f"\n✗ All batches failed!")


if __name__ == '__main__':
    main()

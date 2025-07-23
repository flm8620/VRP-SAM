#!/usr/bin/env python3
"""
VRP-SAM Numpy客户端
使用numpy array格式进行无损图像传输

功能：
1. 直接传输numpy array数据，避免图像压缩损失
2. 支持从各种源加载图像（文件、numpy数组、PIL图像等）
3. 接收numpy格式的分割结果

使用方法:
python inference_client_numpy.py --server_url http://localhost:8080 --class_name cube --query_images img1.jpg img2.jpg
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
import pickle
import zlib


class VRPSAMNumpyClient:
    def __init__(self, server_url):
        self.server_url = server_url.rstrip('/')
        self.session = requests.Session()
        
    def load_image_as_numpy(self, image_source):
        """从各种源加载图像为numpy数组
        
        Args:
            image_source: 可以是文件路径、PIL图像、或numpy数组
            
        Returns:
            numpy数组，形状为(H, W, 3)，数据类型为float32，范围[0, 1]
        """
        if isinstance(image_source, str):
            # 文件路径
            try:
                image = Image.open(image_source).convert('RGB')
                img_array = np.array(image, dtype=np.float32) / 255.0
                return img_array
            except Exception as e:
                print(f"Error loading image {image_source}: {e}")
                return None
                
        elif isinstance(image_source, Image.Image):
            # PIL图像
            image = image_source.convert('RGB')
            img_array = np.array(image, dtype=np.float32) / 255.0
            return img_array
            
        elif isinstance(image_source, np.ndarray):
            # 已经是numpy数组
            img_array = image_source.astype(np.float32)
            
            # 确保范围在[0, 1]
            if img_array.max() > 1.0:
                img_array = img_array / 255.0
                
            # 确保是3通道
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                return img_array
            elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
                return np.repeat(img_array, 3, axis=2)
            elif len(img_array.shape) == 2:
                return np.stack([img_array] * 3, axis=2)
            else:
                print(f"Unsupported array shape: {img_array.shape}")
                return None
        else:
            print(f"Unsupported image source type: {type(image_source)}")
            return None
    
    def numpy_to_compressed_bytes(self, numpy_array):
        """将numpy数组压缩并编码为base64字符串"""
        try:
            # 使用pickle序列化numpy数组
            pickled_data = pickle.dumps(numpy_array)
            # 使用zlib压缩
            compressed_data = zlib.compress(pickled_data)
            # 转为base64
            base64_str = base64.b64encode(compressed_data).decode('utf-8')
            return base64_str
        except Exception as e:
            print(f"Error encoding numpy array: {e}")
            return None
    
    def compressed_bytes_to_numpy(self, base64_str):
        """将base64字符串解压缩为numpy数组"""
        try:
            # 从base64解码
            compressed_data = base64.b64decode(base64_str)
            # 解压缩
            pickled_data = zlib.decompress(compressed_data)
            # 反序列化numpy数组
            numpy_array = pickle.loads(pickled_data)
            return numpy_array
        except Exception as e:
            print(f"Error decoding numpy array: {e}")
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
    
    def predict_with_class_numpy(self, query_images, class_name, use_all_support=None):
        """使用numpy格式进行推理预测
        
        Args:
            query_images: 查询图像列表，可以是文件路径、PIL图像或numpy数组
            class_name: 类别名称
            use_all_support: 是否使用所有支持样本
            
        Returns:
            推理结果，包含numpy格式的mask和概率图
        """
        try:
            print(f"Preparing numpy request for class '{class_name}'...")
            print(f"Query images: {len(query_images)} items")
            
            # 加载并编码查询图像
            query_arrays_encoded = []
            valid_queries = []
            
            for i, img_source in enumerate(query_images):
                print(f"Processing query image {i+1}/{len(query_images)}...")
                img_array = self.load_image_as_numpy(img_source)
                
                if img_array is None:
                    print(f"Failed to load query image {i+1}, skipping")
                    continue
                
                # 压缩编码
                encoded_array = self.numpy_to_compressed_bytes(img_array)
                if encoded_array is None:
                    print(f"Failed to encode query image {i+1}, skipping")
                    continue
                
                query_arrays_encoded.append(encoded_array)
                valid_queries.append(img_source)
                
                # 打印数组信息
                print(f"  Shape: {img_array.shape}, Range: [{img_array.min():.3f}, {img_array.max():.3f}]")
            
            if not query_arrays_encoded:
                print("No valid query images to process")
                return None
            
            # 构建请求
            request_data = {
                'class_name': class_name,
                'query_arrays': query_arrays_encoded,
                'format': 'numpy'  # 标识这是numpy格式请求
            }
            
            if use_all_support is not None:
                request_data['use_all_support'] = use_all_support
            
            print(f"Sending numpy request to server...")
            start_time = time.time()
            
            # 发送请求到numpy专用端点
            response = self.session.post(
                f"{self.server_url}/predict_numpy",
                json=request_data,
                timeout=120
            )
            
            request_time = time.time() - start_time
            print(f"Request completed in {request_time:.3f}s")
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success', False):
                    print(f"✓ Numpy inference successful!")
                    print(f"  Server inference time: {result.get('inference_time', 0):.3f}s")
                    print(f"  Class: {result.get('class_name', 'unknown')}")
                    print(f"  Query count: {result.get('query_count', 0)}")
                    print(f"  Support samples used: {result.get('support_samples_used', 0)}")
                    
                    # 解码numpy结果
                    numpy_results = []
                    for i, res in enumerate(result.get('results', [])):
                        numpy_result = {}
                        
                        # 解码各种结果
                        if 'binary_mask_array' in res:
                            binary_mask = self.compressed_bytes_to_numpy(res['binary_mask_array'])
                            numpy_result['binary_mask'] = binary_mask
                        
                        if 'probability_array' in res:
                            prob_array = self.compressed_bytes_to_numpy(res['probability_array'])
                            numpy_result['probability_map'] = prob_array
                        
                        # 保留统计信息
                        if 'statistics' in res:
                            numpy_result['statistics'] = res['statistics']
                        
                        numpy_results.append(numpy_result)
                        
                        # 打印结果信息
                        if 'binary_mask' in numpy_result:
                            mask = numpy_result['binary_mask']
                            print(f"  Result {i+1}: Binary mask shape {mask.shape}, "
                                  f"positive pixels: {mask.sum():.0f}/{mask.size} ({mask.mean()*100:.1f}%)")
                    
                    # 构建完整结果
                    final_result = {
                        'success': True,
                        'inference_time': result.get('inference_time', 0),
                        'class_name': result.get('class_name', class_name),
                        'query_count': len(numpy_results),
                        'results': numpy_results,
                        'query_sources': valid_queries
                    }
                    
                    return final_result
                else:
                    print(f"✗ Numpy inference failed: {result.get('error', 'Unknown error')}")
                    return None
            else:
                print(f"✗ Server error: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"  Error details: {error_data.get('error', 'Unknown error')}")
                except:
                    print(f"  Response text: {response.text}")
                return None
                
        except Exception as e:
            print(f"✗ Error during numpy prediction: {e}")
            return None
    
    def save_numpy_results(self, result, output_dir):
        """保存numpy格式的推理结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        results = result.get('results', [])
        class_name = result.get('class_name', 'unknown')
        query_sources = result.get('query_sources', [])
        
        saved_files = []
        
        for i, (res, query_source) in enumerate(zip(results, query_sources)):
            # 生成输出文件名
            if isinstance(query_source, str):
                base_name = Path(query_source).stem
            else:
                base_name = f"query_{i+1}"
            
            # 保存numpy数组
            if 'binary_mask' in res:
                mask_path = os.path.join(output_dir, f"{base_name}_binary_mask.npy")
                np.save(mask_path, res['binary_mask'])
                saved_files.append(mask_path)
            
            if 'probability_map' in res:
                prob_path = os.path.join(output_dir, f"{base_name}_probability.npy")
                np.save(prob_path, res['probability_map'])
                saved_files.append(prob_path)
            
            # 同时保存可视化图像（方便查看）
            if 'binary_mask' in res:
                mask = res['binary_mask']
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                img_path = os.path.join(output_dir, f"{base_name}_binary_mask.png")
                mask_img.save(img_path)
                saved_files.append(img_path)
            
            if 'probability_map' in res:
                prob_map = res['probability_map']
                prob_img = Image.fromarray((prob_map * 255).astype(np.uint8))
                img_path = os.path.join(output_dir, f"{base_name}_probability.png")
                prob_img.save(img_path)
                saved_files.append(img_path)
            
            # 保存统计信息
            if 'statistics' in res:
                stats_path = os.path.join(output_dir, f"{base_name}_stats.json")
                with open(stats_path, 'w') as f:
                    json.dump(res['statistics'], f, indent=2)
                saved_files.append(stats_path)
        
        # 保存总体结果信息
        summary_path = os.path.join(output_dir, f"{class_name}_numpy_summary.json")
        summary_data = {k: v for k, v in result.items() if k != 'results'}  # 排除大的numpy数据
        summary_data['result_count'] = len(results)
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        saved_files.append(summary_path)
        
        print(f"Numpy results saved: {len(saved_files)} files in {output_dir}")
        print(f"Saved .npy files for direct numpy loading")
        print(f"Saved .png files for visualization")
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
    parser = argparse.ArgumentParser(description='VRP-SAM Numpy Client - Lossless Image Inference')
    parser.add_argument('--server_url', type=str, default='http://localhost:8080',
                        help='URL of the inference server')
    parser.add_argument('--class_name', type=str, required=True,
                        help='Object class name for inference (e.g., cube, car, table)')
    parser.add_argument('--query_images', nargs='+', required=True,
                        help='Query image files, directories, or patterns')
    parser.add_argument('--output_dir', type=str, default='./numpy_results',
                        help='Output directory for results')
    parser.add_argument('--use_all_support', action='store_true',
                        help='Use all support samples (default: use server config)')
    parser.add_argument('--single_support', action='store_true',
                        help='Use only single support sample')
    parser.add_argument('--max_batch_size', type=int, default=5,
                        help='Maximum number of images to process in one batch')
    
    args = parser.parse_args()
    
    # 创建numpy客户端
    client = VRPSAMNumpyClient(args.server_url)
    
    # 检查服务器状态
    print("=== VRP-SAM Numpy Client ===")
    print("Checking server health...")
    if not client.check_server_health():
        print("Server is not accessible. Please start the server first.")
        return
    
    # 获取可用类别
    print("\nGetting available classes...")
    available_classes = client.get_available_classes()
    
    if not available_classes:
        print("No classes available on server!")
        return
    
    print("Available classes:")
    for class_name, info in available_classes.items():
        print(f"  - {class_name}: {info.get('description', 'No description')} ({info.get('support_count', 0)} samples)")
    
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
    output_dir = Path(args.output_dir) / f"{args.class_name}_numpy_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 批量处理
    total_images = len(query_image_paths)
    batch_size = args.max_batch_size
    total_batches = (total_images + batch_size - 1) // batch_size
    
    success_count = 0
    total_time = 0
    
    print(f"\n=== Starting Numpy Inference ===")
    print(f"Class: {args.class_name}")
    print(f"Total images: {total_images}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches: {total_batches}")
    print(f"Format: Lossless numpy arrays")
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_images)
        batch_paths = query_image_paths[start_idx:end_idx]
        
        print(f"\n--- Batch {batch_idx + 1}/{total_batches} ---")
        print(f"Processing images {start_idx + 1}-{end_idx} of {total_images}")
        
        start_time = time.time()
        
        # 发送numpy推理请求
        result = client.predict_with_class_numpy(
            batch_paths,
            args.class_name,
            use_all_support=use_all_support
        )
        
        if result:
            # 保存结果
            saved_files = client.save_numpy_results(result, str(output_dir))
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
        print(f"\n✓ Numpy processing completed successfully!")
        print(f"Check the results in: {output_dir}")
        print(f"  - .npy files contain raw numpy arrays")
        print(f"  - .png files contain visualizations")
        print(f"  - No compression artifacts!")
    else:
        print(f"\n✗ All batches failed!")


if __name__ == '__main__':
    main()

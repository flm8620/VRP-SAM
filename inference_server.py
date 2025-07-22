#!/usr/bin/env python3
"""
VRP-SAM 推理服务器
提供基于HTTP的VRP-SAM推理服务，支持局域网内的推理请求

功能：
1. 加载VRP-SAM模型
2. 接受HTTP POST请求，包含查询图像和支持样本
3. 返回分割结果和概率热图

使用方法:
python inference_server.py --model_path path/to/model.pt --port 8080

API接口:
POST /predict
- 输入: JSON格式，包含base64编码的图像数据
- 输出: JSON格式，包含分割结果
"""

import os
import json
import base64
import io
import argparse
import time
from threading import Lock
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import cv2

# 导入推理相关模块
from model.VRP_encoder import VRP_encoder
from SAM2pred import SAM_pred
from common import utils


class VRPSAMInferenceServer:
    def __init__(self, model_path, sam_version='vit_h', backbone='resnet50', device='cuda', config_path='support_classes_config.json'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_lock = Lock()  # 确保模型推理的线程安全
        
        # 图像预处理参数
        self.img_size = 512
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        print(f"Initializing VRP-SAM Inference Server...")
        print(f"Using device: {self.device}")
        
        # 加载支持类别配置
        self.config_path = config_path
        self.support_config = self._load_support_config()
        
        # 预加载所有支持样本
        self._preload_support_samples()
        
        # 加载VRP模型
        args = self._create_args(backbone)
        self.vrp_model = VRP_encoder(args, backbone, False)
        
        if model_path and os.path.exists(model_path):
            print(f"Loading VRP model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            checkpoint = checkpoint['model_state_dict']
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
        
        
        print("Server initialization completed!")
        
    def _load_support_config(self):
        """加载支持类别配置"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"Loaded support classes config from {self.config_path}")
                available_classes = list(config['support_classes'].keys())
                print(f"Available classes: {available_classes}")
                return config
            else:
                print(f"Warning: Config file {self.config_path} not found, using empty config")
                return {'support_classes': {}, 'default_settings': {}}
        except Exception as e:
            print(f"Error loading config: {e}")
            return {'support_classes': {}, 'default_settings': {}}
    
    def _preload_support_samples(self):
        """预加载所有支持样本到内存"""
        self.preloaded_samples = {}
        
        for class_name, class_config in self.support_config['support_classes'].items():
            print(f"Preloading support samples for class: {class_name}")
            support_images = []
            support_masks = []
            
            for sample in class_config['support_samples']:
                try:
                    # 加载图像
                    img_path = sample['image']
                    mask_path = sample['mask']
                    
                    if os.path.exists(img_path) and os.path.exists(mask_path):
                        img = Image.open(img_path).convert('RGB')
                        mask = Image.open(mask_path).convert('RGB')
                        
                        # 预处理并存储tensor
                        img_tensor = self.preprocess_image(img)
                        mask_tensor = self.preprocess_mask(mask)
                        
                        support_images.append(img_tensor)
                        support_masks.append(mask_tensor)
                        print(f"  Loaded: {os.path.basename(img_path)}")
                    else:
                        print(f"  Warning: Missing files for {img_path} or {mask_path}")
                except Exception as e:
                    print(f"  Error loading sample {sample}: {e}")
            
            if support_images:
                # 合并所有支持样本为batch tensor
                self.preloaded_samples[class_name] = {
                    'images': torch.cat(support_images, dim=0),
                    'masks': torch.cat(support_masks, dim=0),
                    'count': len(support_images)
                }
                print(f"  Successfully preloaded {len(support_images)} samples for {class_name}")
            else:
                print(f"  No valid samples found for {class_name}")
    
    def get_available_classes(self):
        """获取可用的识别类别"""
        return list(self.support_config['support_classes'].keys())
    
    def _create_args(self, backbone):
        """创建模型需要的参数"""
        class Args:
            def __init__(self):
                self.condition = 'mask'
                self.backbone = backbone
                self.num_query = 50
                
        return Args()
    
    def base64_to_image(self, base64_str):
        """将base64字符串转换为PIL图像"""
        try:
            # 移除可能的前缀
            if ',' in base64_str:
                base64_str = base64_str.split(',')[1]
            
            image_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            return image
        except Exception as e:
            raise ValueError(f"Invalid base64 image data: {e}")
    
    def image_to_base64(self, image):
        """将PIL图像转换为base64字符串"""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return image_base64
    
    def preprocess_image(self, image):
        """预处理图像"""
        # 调整大小
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_array = np.array(image) / 255.0
        
        # 归一化
        for i in range(3):
            img_array[:, :, i] = (img_array[:, :, i] - self.mean[i]) / self.std[i]
        
        # 转为tensor
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        return img_tensor.unsqueeze(0)  # 添加batch维度
    
    def preprocess_mask(self, mask_image):
        """预处理mask图像"""
        mask = mask_image.resize((self.img_size, self.img_size), Image.NEAREST)
        mask_array = np.array(mask)
        
        # 如果是RGB图像，转换为灰度
        if len(mask_array.shape) == 3:
            mask_array = np.mean(mask_array, axis=2)
        
        # 二值化
        mask_array = (mask_array > 128).astype(np.float32)
        
        # 转为tensor
        mask_tensor = torch.from_numpy(mask_array).float()
        return mask_tensor.unsqueeze(0)  # 添加batch维度
    
    def predict_with_class(self, query_images, class_name, use_all_support=None):
        """使用预配置类别执行推理预测"""
        with self.model_lock:
            # 检查类别是否存在
            if class_name not in self.preloaded_samples:
                raise ValueError(f"Class '{class_name}' not found. Available classes: {self.get_available_classes()}")
            
            # 获取默认设置
            default_settings = self.support_config.get('default_settings', {})
            if use_all_support is None:
                use_all_support = default_settings.get('use_all_support', True)
            
            # 获取预加载的支持样本
            support_data = self.preloaded_samples[class_name]
            support_imgs = support_data['images'].to(self.device)
            support_masks = support_data['masks'].to(self.device)
            
            results = []
            
            for query_image in query_images:
                # 预处理查询图像
                if isinstance(query_image, str):
                    # 如果是base64字符串，解码
                    query_image = self.base64_to_image(query_image)
                
                query_tensor = self.preprocess_image(query_image).to(self.device)
                query_name = ["query"]
                
                with torch.no_grad():
                    if use_all_support and len(support_imgs) > 1:
                        # 使用多个支持样本
                        protos_list = []
                        
                        for i in range(len(support_imgs)):
                            support_img = support_imgs[i:i+1]
                            support_mask = support_masks[i:i+1]
                            
                            protos, _ = self.vrp_model(
                                'mask',
                                query_tensor,
                                support_img,
                                support_mask,
                                training=False
                            )
                            protos_list.append(protos)
                        
                        # 合并多个prototypes
                        combined_protos = torch.cat(protos_list, dim=1)
                    else:
                        # 只使用第一个支持样本
                        support_img = support_imgs[0:1]
                        support_mask = support_masks[0:1]
                        
                        combined_protos, _ = self.vrp_model(
                            'mask',
                            query_tensor,
                            support_img,
                            support_mask,
                            training=False
                        )
                    
                    # SAM预测
                    low_masks, pred_mask = self.sam_model(query_tensor, query_name, combined_protos)
                    
                    # 获取概率分布
                    prob_mask = torch.sigmoid(low_masks)
                    
                    # 二值化mask
                    binary_mask = prob_mask > 0.5
                    binary_mask = binary_mask.float()
                
                results.append((binary_mask.squeeze().cpu().numpy(), prob_mask.squeeze().cpu().numpy()))
            
            return results
    
    
    def create_visualization(self, query_image, pred_mask, prob_mask):
        """创建可视化结果"""
        # 调整查询图像大小以匹配mask
        query_resized = query_image.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_array = np.array(query_resized)
        
        # 创建二值mask可视化
        binary_vis = Image.fromarray((pred_mask * 255).astype(np.uint8))
        
        # 创建概率热图
        prob_heatmap = (prob_mask * 255).astype(np.uint8)
        prob_vis = Image.fromarray(prob_heatmap)
        
        # 创建叠加图像
        mask_colored = np.zeros_like(img_array)
        mask_colored[:, :, 0] = pred_mask * 255  # 红色通道
        
        alpha = 0.5
        overlay = img_array * (1 - alpha) + mask_colored * alpha
        overlay_vis = Image.fromarray(overlay.astype(np.uint8))
        
        return binary_vis, prob_vis, overlay_vis


# Flask应用
app = Flask(__name__)
inference_server = None


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'device': str(inference_server.device)
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 解析请求
        data = request.json
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        return predict_with_class(data)
            
    except Exception as e:
        print(f"Error during inference: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': time.time()
        }), 500


def predict_with_class(data):
    """基于预配置类别的推理"""
    # 验证必需字段
    if 'class_name' not in data:
        return jsonify({'error': 'Missing required field: class_name'}), 400
    if 'query_images' not in data:
        return jsonify({'error': 'Missing required field: query_images'}), 400
    
    class_name = data['class_name']
    query_images_data = data['query_images']
    
    # 检查类别是否存在
    available_classes = inference_server.get_available_classes()
    if class_name not in available_classes:
        return jsonify({
            'error': f'Unknown class: {class_name}',
            'available_classes': available_classes
        }), 400
    
    # 解码查询图像
    query_images = []
    for img_data in query_images_data:
        query_images.append(inference_server.base64_to_image(img_data))
    
    # 获取推理参数
    use_all_support = data.get('use_all_support', None)  # None表示使用默认值
    
    # 执行推理
    start_time = time.time()
    results = inference_server.predict_with_class(query_images, class_name, use_all_support)
    inference_time = time.time() - start_time
    
    # 处理结果
    response_results = []
    
    for i, (pred_mask, prob_mask) in enumerate(results):
        # 创建可视化结果
        binary_vis, prob_vis, overlay_vis = inference_server.create_visualization(
            query_images[i], pred_mask, prob_mask
        )
        
        # 计算统计信息
        stats = {
            'mean_probability': float(prob_mask.mean()),
            'std_probability': float(prob_mask.std()),
            'high_confidence_ratio': float((prob_mask > 0.8).sum() / prob_mask.size),
            'low_confidence_ratio': float((prob_mask < 0.2).sum() / prob_mask.size),
            'uncertain_ratio': float(((prob_mask > 0.3) & (prob_mask < 0.7)).sum() / prob_mask.size),
            'positive_pixels': int((pred_mask > 0.5).sum()),
            'total_pixels': int(pred_mask.size)
        }
        
        result_item = {
            'statistics': stats,
            'results': {
                'binary_mask': inference_server.image_to_base64(binary_vis),
                'probability_heatmap': inference_server.image_to_base64(prob_vis),
                'overlay': inference_server.image_to_base64(overlay_vis)
            }
        }
        response_results.append(result_item)
    
    # 构建响应
    response = {
        'success': True,
        'timestamp': time.time(),
        'inference_time': inference_time,
        'class_name': class_name,
        'query_count': len(query_images),
        'support_samples_used': inference_server.preloaded_samples[class_name]['count'],
        'results': response_results
    }
    
    print(f"Class-based inference completed in {inference_time:.3f}s for {len(query_images)} images of class '{class_name}'")
    
    return jsonify(response)


@app.route('/classes', methods=['GET'])
def get_classes():
    """获取可用的识别类别"""
    try:
        classes = inference_server.get_available_classes()
        class_info = {}
        
        for class_name in classes:
            class_config = inference_server.support_config['support_classes'][class_name]
            class_info[class_name] = {
                'description': class_config.get('description', ''),
                'support_count': len(class_config['support_samples'])
            }
        
        return jsonify({
            'success': True,
            'classes': class_info,
            'total_classes': len(classes)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/info', methods=['GET'])
def server_info():
    """获取服务器信息"""
    classes = inference_server.get_available_classes()
    return jsonify({
        'model_device': str(inference_server.device),
        'image_size': inference_server.img_size,
        'available_endpoints': ['/health', '/predict', '/info', '/classes'],
        'available_classes': classes,
        'version': '2.0.0'
    })


def main():
    parser = argparse.ArgumentParser(description='VRP-SAM Inference Server')
    parser.add_argument('--model_path', type=str, 
                        default='/home/leman.feng/VRP-SAM-eval/logs/trn1_coco_mask_fold0.log/best_model.pt',
                        help='Path to trained VRP model checkpoint')
    parser.add_argument('--config_path', type=str, default='support_classes_config.json',
                        help='Path to support classes configuration file')
    parser.add_argument('--sam_version', type=str, default='vit_h',
                        choices=['vit_h', 'vit_l'], help='SAM model version')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['vgg16', 'resnet50', 'resnet101'], help='VRP backbone')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for inference')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=8080,
                        help='Port to bind the server to')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    
    args = parser.parse_args()
    
    # 设置随机种子
    utils.fix_randseed(0)
    
    # 创建推理服务器实例
    global inference_server
    inference_server = VRPSAMInferenceServer(
        model_path=args.model_path,
        sam_version=args.sam_version,
        backbone=args.backbone,
        device=args.device,
        config_path=args.config_path
    )
    
    print(f"\nStarting VRP-SAM Inference Server...")
    print(f"Server will be available at: http://{args.host}:{args.port}")
    print(f"Health check: http://{args.host}:{args.port}/health")
    print(f"Prediction endpoint: http://{args.host}:{args.port}/predict")
    print(f"Classes endpoint: http://{args.host}:{args.port}/classes")
    print(f"Server info: http://{args.host}:{args.port}/info")
    print(f"\nPress Ctrl+C to stop the server")
    
    # 启动Flask服务器
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True  # 启用多线程支持
    )


if __name__ == '__main__':
    main()

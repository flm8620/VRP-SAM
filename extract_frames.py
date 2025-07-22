#!/usr/bin/env python3
# extract_frames.py
import cv2
import argparse
import numpy as np
from pathlib import Path

def extract_frames_at_times(video_path, timestamps, output_dir="extracted_frames"):
    """
    从视频中提取指定时间点的帧
    
    Args:
        video_path: 视频文件路径
        timestamps: 时间点列表（秒）
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"视频信息:")
    print(f"  文件: {video_path}")
    print(f"  帧率: {fps:.2f} FPS")
    print(f"  总帧数: {total_frames}")
    print(f"  时长: {duration:.2f} 秒")
    print(f"  输出目录: {output_dir}")
    print()

    # extract video name stem
    video_path = Path(video_path)
    video_name_stem = video_path.stem

    extracted_count = 0
    
    for i, timestamp in enumerate(timestamps):
        if timestamp > duration:
            print(f"⚠️  时间点 {timestamp:.2f}s 超出视频时长 {duration:.2f}s，跳过")
            continue
        
        # 计算最接近的帧号
        target_frame = int(timestamp * fps)
        
        # 设置视频位置到目标帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️  无法读取时间点 {timestamp:.2f}s 的帧，跳过")
            continue
        
        # 生成输出文件名
        output_filename = f"{video_name_stem}_t{timestamp:.2f}s_f{target_frame:06d}.png"
        output_path = output_dir / output_filename
        
        # 保存帧
        cv2.imwrite(str(output_path), frame)
        
        # 获取实际时间戳
        actual_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # -1因为read()后位置会前进
        actual_timestamp = actual_frame / fps
        
        print(f"✓ 提取帧 {i+1}: 目标={timestamp:.2f}s, 实际={actual_timestamp:.2f}s, "
              f"帧号={actual_frame}, 文件={output_filename}")
        
        extracted_count += 1
    
    cap.release()
    print(f"\n完成! 共提取 {extracted_count} 帧")

def parse_timestamps(timestamp_str):
    """解析时间戳字符串"""
    timestamps = []
    for part in timestamp_str.split(','):
        part = part.strip()
        if '-' in part:
            # 范围格式: "10-20" 表示从10秒到20秒
            start, end = map(float, part.split('-'))
            # 默认每秒一帧
            timestamps.extend(np.arange(start, end + 1, 1.0))
        else:
            # 单个时间点
            timestamps.append(float(part))
    
    return sorted(list(set(timestamps)))  # 去重并排序

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从视频中提取指定时间点的帧")
    parser.add_argument("video", help="输入视频文件路径")
    parser.add_argument("timestamps", help="时间戳，支持格式: '1.5,3.0,5.5' 或 '10-15,20,25-30'")
    parser.add_argument("-o", "--output", default="extracted_frames", 
                       help="输出目录 (默认: extracted_frames)")
    
    args = parser.parse_args()
    
    try:
        timestamps = parse_timestamps(args.timestamps)
        print(f"解析到的时间点: {timestamps}")
        print()
        
        extract_frames_at_times(args.video, timestamps, args.output)
        
    except Exception as e:
        print(f"错误: {e}")
        exit(1)

#!/usr/bin/env python3
"""
Python进程管理器
功能：显示Python进程的完整树状结构，统计内存使用，并支持按树分支选择性终止
特性：支持查看当前用户或所有用户的Python进程

依赖:
- 推荐安装: pip install nvidia-ml-py (用于更好的GPU监控)
- 备用方案: nvidia-smi命令行工具
"""

import os
import sys
import subprocess
import re
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import pynvml


@dataclass
class ProcessInfo:
    """进程信息数据类"""
    pid: int
    ppid: int
    pmem: float
    vsz: int  # KB
    rss: int  # KB
    etime: str
    cmd: str
    username: str = ""  # 进程所属用户名
    is_python: bool = False
    gpu_memory: int = 0  # MB
    children: List['ProcessInfo'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


class PythonProcessManager:
    """Python进程管理器"""
    
    def __init__(self, all_users: bool = False):
        self.username = os.getenv('USER', 'unknown')
        self.all_users = all_users
        self.processes: Dict[int, ProcessInfo] = {}
        self.python_pids: Set[int] = set()
        self.tree_roots: List[ProcessInfo] = []
        self.current_tree_info: List = []  # 保存当前显示的进程树信息和顺序
        user_suffix = "all_users" if all_users else self.username
        self.cmd_dump_file = f"python_processes_commands_{user_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.gpu_memory_map: Dict[int, int] = {}  # PID -> GPU memory usage in MB
        
    def get_gpu_memory_usage(self) -> Dict[int, int]:
        """获取所有进程的GPU内存使用情况"""
        gpu_memory_map = {}
        
        # 初始化NVML
        pynvml.nvmlInit()
        
        # 获取GPU数量
        device_count = pynvml.nvmlDeviceGetCount()
        
        # 遍历所有GPU设备
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # 获取该GPU上运行的进程
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                for proc in procs:
                    pid = proc.pid
                    # 内存使用量已经是字节，转换为MB
                    memory_mb = proc.usedGpuMemory // (1024 * 1024)
                    
                    # 如果进程在多个GPU上运行，累加内存使用
                    if pid in gpu_memory_map:
                        gpu_memory_map[pid] += memory_mb
                    else:
                        gpu_memory_map[pid] = memory_mb
                        
            except pynvml.NVMLError:
                # 某些GPU可能不支持查询运行进程
                continue
        
        # 清理NVML
        pynvml.nvmlShutdown()
            
        
        return gpu_memory_map
    
        """备用方案：使用命令行获取GPU内存使用情况"""
        gpu_memory_map = {}
        
        try:
            # 使用nvidia-smi获取GPU进程信息
            result = subprocess.run(['nvidia-smi', 'pmon', '-c', '1', '-s', 'um'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.startswith('#') or not line.strip():
                        continue
                    
                    # 解析nvidia-smi pmon输出格式: gpu_id pid type sm mem enc dec command
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            pid = int(parts[1])
                            mem_usage = parts[5]  # 内存使用量
                            
                            # 解析内存使用量（可能是数字或'-'）
                            if mem_usage != '-' and mem_usage.isdigit():
                                gpu_memory_map[pid] = int(mem_usage)
                        except (ValueError, IndexError):
                            continue
            
            # 如果pmon不工作，尝试使用另一种方法
            if not gpu_memory_map:
                result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', 
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if line.strip():
                            try:
                                parts = line.split(',')
                                if len(parts) >= 2:
                                    pid = int(parts[0].strip())
                                    memory_mb = int(parts[1].strip())
                                    gpu_memory_map[pid] = memory_mb
                            except (ValueError, IndexError):
                                continue
                                
        except subprocess.TimeoutExpired:
            print("⚠️ nvidia-smi 命令超时，跳过GPU内存统计")
        except FileNotFoundError:
            print("⚠️ 未找到nvidia-smi命令，跳过GPU内存统计")
        except Exception as e:
            print(f"⚠️ 获取GPU内存使用失败: {e}")
        
        return gpu_memory_map
    
    def get_python_processes(self) -> List[int]:
        """获取Python进程的PID列表"""
        try:
            current_pid = os.getpid()  # 获取当前进程PID
            
            if self.all_users:
                # 获取所有用户的Python进程
                result = subprocess.run(['pgrep', '-f', '^python'], 
                                      capture_output=True, text=True)
            else:
                # 只获取当前用户的Python进程
                result = subprocess.run(['pgrep', '-u', self.username, '-f', 'python'], 
                                      capture_output=True, text=True)
            
            if result.returncode == 0:
                pids = [int(pid) for pid in result.stdout.strip().split('\n') if pid]
                # 排除当前进程
                return [pid for pid in pids if pid != current_pid]
            return []
        except Exception as e:
            print(f"❌ 获取Python进程失败: {e}")
            return []
    
    def get_process_chain(self, pid: int, visited: Set[int] = None) -> Set[int]:
        """递归获取进程链中的所有相关进程"""
        if visited is None:
            visited = set()
        
        if pid in visited or pid <= 1:
            return visited
        
        visited.add(pid)
        
        try:
            # 获取父进程PID
            result = subprocess.run(['ps', '-o', 'ppid=', '-p', str(pid)], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                ppid = int(result.stdout.strip())
                if ppid > 1:
                    self.get_process_chain(ppid, visited)
        except (ValueError, subprocess.SubprocessError):
            pass
        
        return visited
    
    def collect_process_info(self) -> bool:
        """收集所有相关进程信息"""
        print("🔍 正在收集进程树结构...")
        
        # 获取Python进程列表
        python_pids = self.get_python_processes()
        if not python_pids:
            print("✅ 未找到任何以 'python' 开头的进程。")
            return False
        
        self.python_pids = set(python_pids)
        
        # 获取GPU内存使用情况
        print("🎮 正在获取GPU内存使用情况...")
        self.gpu_memory_map = self.get_gpu_memory_usage()
        
        # 收集所有相关进程ID
        all_relevant_pids = set()
        for pid in python_pids:
            all_relevant_pids.update(self.get_process_chain(pid))
        
        # 获取所有相关进程的详细信息
        if all_relevant_pids:
            try:
                pids_str = ' '.join(map(str, sorted(all_relevant_pids)))
                # 在所有用户模式下，需要获取用户信息
                if self.all_users:
                    ps_format = 'pid,ppid,pmem,vsz,rss,etime,user,cmd'
                else:
                    ps_format = 'pid,ppid,pmem,vsz,rss,etime,cmd'
                
                result = subprocess.run(['ps', '-ww', '-o', ps_format,
                                       '--no-headers'] + list(map(str, sorted(all_relevant_pids))),
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.parse_process_output(result.stdout)
                    self.build_process_tree()
                    return True
                else:
                    print(f"❌ 获取进程信息失败: {result.stderr}")
                    return False
            except Exception as e:
                print(f"❌ 处理进程信息时出错: {e}")
                return False
        
        return False
    
    def parse_process_output(self, output: str):
        """解析ps命令输出"""
        for line in output.strip().split('\n'):
            if not line.strip():
                continue
            
            if self.all_users:
                # 包含用户信息的格式: pid ppid pmem vsz rss etime user cmd
                parts = line.strip().split(None, 7)  # 分割成8部分
                if len(parts) >= 7:
                    try:
                        pid = int(parts[0])
                        ppid = int(parts[1])
                        pmem = float(parts[2])
                        vsz = int(parts[3])
                        rss = int(parts[4])
                        etime = parts[5]
                        username = parts[6]
                        cmd = parts[7] if len(parts) > 7 else ""
                        
                        # 获取GPU内存使用
                        gpu_memory = self.gpu_memory_map.get(pid, 0)
                        
                        process_info = ProcessInfo(
                            pid=pid, ppid=ppid, pmem=pmem, vsz=vsz, 
                            rss=rss, etime=etime, cmd=cmd, username=username,
                            is_python=(pid in self.python_pids),
                            gpu_memory=gpu_memory
                        )
                        self.processes[pid] = process_info
                    except (ValueError, IndexError) as e:
                        print(f"⚠️ 解析进程信息失败: {line} - {e}")
                        continue
            else:
                # 不包含用户信息的格式: pid ppid pmem vsz rss etime cmd
                parts = line.strip().split(None, 6)  # 分割成7部分
                if len(parts) >= 6:
                    try:
                        pid = int(parts[0])
                        ppid = int(parts[1])
                        pmem = float(parts[2])
                        vsz = int(parts[3])
                        rss = int(parts[4])
                        etime = parts[5]
                        cmd = parts[6] if len(parts) > 6 else ""
                        
                        # 获取GPU内存使用
                        gpu_memory = self.gpu_memory_map.get(pid, 0)
                        
                        process_info = ProcessInfo(
                            pid=pid, ppid=ppid, pmem=pmem, vsz=vsz, 
                            rss=rss, etime=etime, cmd=cmd, username=self.username,
                            is_python=(pid in self.python_pids),
                            gpu_memory=gpu_memory
                        )
                        self.processes[pid] = process_info
                    except (ValueError, IndexError) as e:
                        print(f"⚠️ 解析进程信息失败: {line} - {e}")
                        continue
    
    def build_process_tree(self):
        """构建进程树结构"""
        # 建立父子关系
        for pid, process in self.processes.items():
            if process.ppid in self.processes:
                self.processes[process.ppid].children.append(process)
        
        # 找到根进程（父进程不在我们的相关进程列表中）
        for pid, process in self.processes.items():
            if process.ppid not in self.processes:
                self.tree_roots.append(process)
    
    def calculate_subtree_python_memory(self, root: ProcessInfo) -> Tuple[float, int, int, int, int]:
        """计算子树中所有Python进程的内存使用和整个子树的GPU内存使用"""
        total_pmem = 0.0
        total_rss = 0
        total_vsz = 0
        total_gpu_memory = 0
        python_count = 0
        
        def traverse(process: ProcessInfo):
            nonlocal total_pmem, total_rss, total_vsz, python_count, total_gpu_memory
            
            # 累加所有进程的GPU内存
            total_gpu_memory += process.gpu_memory
            
            if process.is_python:
                total_pmem += process.pmem
                total_rss += process.rss
                total_vsz += process.vsz
                python_count += 1
            
            for child in process.children:
                traverse(child)
        
        traverse(root)
        return total_pmem, total_vsz, total_rss, total_gpu_memory, python_count
    
    def format_memory(self, size_kb: int) -> str:
        """格式化内存大小"""
        if size_kb >= 1048576:  # >= 1GB
            return f"{size_kb / 1048576:.1f}GB"
        elif size_kb >= 1024:   # >= 1MB
            return f"{size_kb / 1024:.1f}MB"
        else:
            return f"{size_kb}KB"
    
    def format_uptime(self, etime: str) -> str:
        """格式化运行时间"""
        # etime 格式可能是：MM:SS, HH:MM:SS, 或 DD-HH:MM:SS
        if re.match(r'^\d+-\d+:\d+:\d+$', etime):
            # DD-HH:MM:SS 格式
            days, time_part = etime.split('-', 1)
            return f"{days}天 {time_part}"
        elif re.match(r'^\d+:\d+:\d+$', etime):
            # HH:MM:SS 格式
            return etime
        elif re.match(r'^\d+:\d+$', etime):
            # MM:SS 格式
            return f"00:{etime}"
        else:
            return etime
    
    def print_process_tree(self, process: ProcessInfo, prefix: str = "", 
                          is_last: bool = True, depth: int = 0, cmd_file=None):
        """递归打印进程树"""
        if depth > 10:  # 防止过深递归
            return
        
        # 确定显示符号和颜色
        if process.is_python:
            python_mark = "🐍"
            color_start = "\033[1;32m"  # 绿色加粗
        else:
            python_mark = "📁"
            color_start = "\033[1;36m"  # 青色加粗
        color_end = "\033[0m"
        
        # 构建树形前缀
        tree_char = "└── " if is_last else "├── "
        
        # 构建GPU内存显示
        gpu_display = ""
        if process.gpu_memory > 0:
            gpu_display = f" GPU {self.format_memory(process.gpu_memory * 1024):<8}"
        else:
            gpu_display = f" GPU {'0MB':<8}"
        
        # 构建用户显示信息
        user_display = ""
        if self.all_users and process.username:
            user_display = f" 👤{process.username:<8}"
        
        # 打印进程基本信息
        print(f"{color_start}{prefix}{tree_char}{python_mark} PID {process.pid:<8} "
              f"RSS {self.format_memory(process.rss):<8} "
              f"VSZ {self.format_memory(process.vsz):<8} "
              f"%MEM {process.pmem:<5.1f}%{gpu_display}{user_display} "
              f"运行时间 {self.format_uptime(process.etime):<12}{color_end}")
        
        # 写入完整命令行到文件
        if cmd_file:
            gpu_info = f" (GPU: {process.gpu_memory}MB)" if process.gpu_memory > 0 else ""
            cmd_file.write(f"PID {process.pid} (运行时间: {self.format_uptime(process.etime)}{gpu_info}): {process.cmd}\n")
        
        # 打印命令行（可能截断）
        cmd_prefix = prefix + ("    " if is_last else "│   ")
        max_cmd_length = 120
        display_cmd = process.cmd
        
        if len(process.cmd) > max_cmd_length:
            display_cmd = process.cmd[:max_cmd_length] + "..."
        
        print(f"\033[2m{cmd_prefix}    📝 {display_cmd}\033[0m")
        
        # 递归打印子进程
        for i, child in enumerate(process.children):
            child_is_last = (i == len(process.children) - 1)
            new_prefix = prefix + ("    " if is_last else "│   ")
            self.print_process_tree(child, new_prefix, child_is_last, depth + 1, cmd_file)
    
    def get_subtree_python_pids(self, root: ProcessInfo) -> List[int]:
        """获取子树中所有Python进程的PID"""
        python_pids = []
        
        def traverse(process: ProcessInfo):
            if process.is_python:
                python_pids.append(process.pid)
            for child in process.children:
                traverse(child)
        
        traverse(root)
        return python_pids
    
    def get_subtree_all_pids(self, root: ProcessInfo) -> List[int]:
        """获取子树中所有进程的PID（包括Python和非Python进程）"""
        all_pids = []
        
        def traverse(process: ProcessInfo):
            all_pids.append(process.pid)
            for child in process.children:
                traverse(child)
        
        traverse(root)
        return all_pids
    
    def find_process_by_pid(self, pid: int) -> Optional[ProcessInfo]:
        """通过PID查找进程"""
        return self.processes.get(pid)
    
    def get_pids_from_subtrees(self, target_pids: List[int], kill_all_processes: bool = False) -> List[int]:
        """获取指定PID节点及其所有子进程的PID列表"""
        result_pids = []
        
        for target_pid in target_pids:
            process = self.find_process_by_pid(target_pid)
            if process:
                if kill_all_processes:
                    # 获取所有进程（Python和非Python）
                    subtree_pids = self.get_subtree_all_pids(process)
                else:
                    # 只获取Python进程
                    subtree_pids = self.get_subtree_python_pids(process)
                result_pids.extend(subtree_pids)
            else:
                print(f"⚠️ 警告: 未找到PID {target_pid}")
        
        return sorted(list(set(result_pids)))
    
    def display_process_trees(self):
        """显示所有进程树"""
        if not self.tree_roots:
            print("❌ 没有找到进程树根节点")
            return
        
        # 计算每个树的Python进程内存使用并排序
        tree_info = []
        for root in self.tree_roots:
            pmem, vsz, rss, gpu_memory, python_count = self.calculate_subtree_python_memory(root)
            python_pids = self.get_subtree_python_pids(root)
            tree_info.append((root, pmem, vsz, rss, gpu_memory, python_count, python_pids))
        
        # 按RSS排序（降序）
        tree_info.sort(key=lambda x: x[3], reverse=True)
        
        # 保存当前显示的树信息
        self.current_tree_info = tree_info
        
        print("\n🌳 Python进程树结构分析结果（按Python进程内存使用排序）：")
        print("=" * 70)
        user_scope = "所有用户" if self.all_users else f"用户 {self.username}"
        print(f"📋 查看范围: {user_scope} 的Python进程")
        if self.all_users:
            print("图例: 🐍=Python进程  📁=其他进程  GPU=显存使用(MB)  👤=用户名")
        else:
            print("图例: �=Python进程  📁=其他进程  GPU=显存使用(MB)")
        print("�💡 进程树编号用于快速选择，记住编号可以使用 '编号1-编号2' 格式批量操作\n")
        
        # 创建命令行导出文件
        with open(self.cmd_dump_file, 'w', encoding='utf-8') as cmd_file:
            cmd_file.write(f"Python进程完整命令行导出文件\n")
            cmd_file.write(f"生成时间: {datetime.now()}\n")
            cmd_file.write(f"生成者: {self.username}\n")
            cmd_file.write(f"查看范围: {user_scope}\n")
            cmd_file.write("=" * 40 + "\n\n")
            
            if self.all_users:
                # 所有用户模式：按用户分组显示
                self._display_by_users(tree_info, cmd_file)
            else:
                # 单用户模式：直接显示
                self._display_single_user(tree_info, cmd_file)
            
            # 写入总结信息
            cmd_file.write(f"\n{'='*40}\n")
            cmd_file.write("总结:\n")
            cmd_file.write(f"- 总共发现 {len(tree_info)} 个进程树\n")
            cmd_file.write(f"- 总共发现 {len(self.python_pids)} 个Python进程\n")
            cmd_file.write(f"- 分析完成时间: {datetime.now()}\n")
        
        print(f"\n📄 完整命令行已导出到文件: {self.cmd_dump_file}")
        print(f"💡 使用 'cat {self.cmd_dump_file}' 或 'less {self.cmd_dump_file}' 查看完整命令行")
        
        return tree_info
    
    def _display_by_users(self, tree_info, cmd_file):
        """按用户分组显示进程树"""
        # 按用户分组进程树
        user_trees = defaultdict(list)
        for i, (root, pmem, vsz, rss, gpu_memory, python_count, python_pids) in enumerate(tree_info):
            user = root.username if root.username else "unknown"
            user_trees[user].append((i+1, root, pmem, vsz, rss, gpu_memory, python_count, python_pids))
        
        # 按用户名排序
        sorted_users = sorted(user_trees.keys())
        
        tree_counter = 1
        for user in sorted_users:
            user_tree_list = user_trees[user]
            
            # 计算该用户的总统计
            user_python_count = sum(python_count for _, _, _, _, _, _, python_count, _ in user_tree_list)
            user_total_rss = sum(rss for _, _, _, _, rss, _, _, _ in user_tree_list)
            user_total_vsz = sum(vsz for _, _, _, _, _, vsz, _, _ in user_tree_list)
            user_total_pmem = sum(pmem for _, _, pmem, _, _, _, _, _ in user_tree_list)
            user_total_gpu_mem = sum(gpu_memory for _, _, _, _, _, gpu_memory, _, _ in user_tree_list)
            
            print(f"\n{'🧑‍💻' if user != 'unknown' else '❓'} 用户: {user}")
            print(f"📊 用户统计: {len(user_tree_list)}个进程树, {user_python_count}个Python进程")
            print(f"💾 用户内存: RSS={self.format_memory(user_total_rss)}, VSZ={self.format_memory(user_total_vsz)}, %MEM={user_total_pmem:.1f}%, GPU={self.format_memory(user_total_gpu_mem * 1024)}")
            print("=" * 68)
            
            # 写入用户分组信息到文件
            cmd_file.write(f"\n{'='*50}\n")
            cmd_file.write(f"用户: {user}\n")
            cmd_file.write(f"统计: {len(user_tree_list)}个进程树, {user_python_count}个Python进程\n")
            cmd_file.write(f"内存: RSS={self.format_memory(user_total_rss)}, VSZ={self.format_memory(user_total_vsz)}\n")
            cmd_file.write(f"{'='*50}\n")
            
            for original_index, root, pmem, vsz, rss, gpu_memory, python_count, python_pids in user_tree_list:
                print(f"🌲 进程树 {tree_counter}: 根进程 PID {root.pid} (用户: {user})")
                print(f"📊 Python进程统计: {python_count}个进程, "
                      f"总内存 RSS={self.format_memory(rss)}, "
                      f"VSZ={self.format_memory(vsz)}, %MEM={pmem:.1f}%, "
                      f"GPU={self.format_memory(gpu_memory * 1024)}")
                print(f"🐍 Python进程列表: {' '.join(map(str, python_pids))}")
                print()
                
                # 写入文件分组信息
                cmd_file.write(f"\n进程树 {tree_counter}: 根进程 PID {root.pid} (用户: {user})\n")
                cmd_file.write(f"Python进程统计: {python_count}个进程, 总内存 RSS={self.format_memory(rss)}\n")
                cmd_file.write("-" * 33 + "\n")
                
                # 打印进程树
                self.print_process_tree(root, "", True, 0, cmd_file)
                print()
                
                tree_counter += 1
    
    def _display_single_user(self, tree_info, cmd_file):
        """单用户模式显示进程树"""
        for i, (root, pmem, vsz, rss, gpu_memory, python_count, python_pids) in enumerate(tree_info):
            print("=" * 68)
            print(f"🌲 进程树 {i+1}: 根进程 PID {root.pid}")
            print(f"📊 Python进程统计: {python_count}个进程, "
                  f"总内存 RSS={self.format_memory(rss)}, "
                  f"VSZ={self.format_memory(vsz)}, %MEM={pmem:.1f}%, "
                  f"GPU={self.format_memory(gpu_memory * 1024)}")
            print(f"🐍 Python进程列表: {' '.join(map(str, python_pids))}")
            print()
            
            # 写入文件分组信息
            cmd_file.write(f"\n{'='*33}\n")
            cmd_file.write(f"进程树 {i+1}: 根进程 PID {root.pid}\n")
            cmd_file.write(f"Python进程统计: {python_count}个进程, 总内存 RSS={self.format_memory(rss)}\n")
            cmd_file.write(f"{'='*33}\n")
            
            # 打印进程树
            self.print_process_tree(root, "", True, 0, cmd_file)
            print()
    
    def interactive_kill(self):
        """交互式选择要终止的进程"""
        print("\n🎯 请选择终止方式:")
        print("1. 按进程PID终止")
        print("2. 按进程树编号终止")
        
        try:
            choice = input("请选择 (1/2): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n🚫 操作已取消，未终止任何进程。")
            return
        
        if choice == '1':
            self.kill_by_pids()
        elif choice == '2':
            self.kill_by_tree_numbers()
        else:
            print("❌ 无效选择，操作已取消。")
    
    def kill_by_tree_numbers(self):
        """按进程树编号终止进程"""
        print("\n🌲 请输入要终止的进程树编号:")
        print("💡 支持格式:")
        print("   - 单个编号: 1")
        print("   - 多个编号: 1 3 5")
        print("   - 范围: 1-5 (包含1和5)")
        print("   - 混合: 1 3-5 8 10-12")
        print("💡 空白回车取消操作")
        
        try:
            tree_input = input("输入编号: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n🚫 操作已取消，未终止任何进程。")
            return
        
        if not tree_input:
            print("🚫 操作已取消，未终止任何进程。")
            return
        
        # 解析树编号范围
        try:
            tree_numbers = self.parse_tree_numbers(tree_input)
        except ValueError as e:
            print(f"⚠️ 输入格式错误: {e}")
            return
        
        if not tree_numbers:
            print("🚫 没有输入有效的编号，操作已取消。")
            return
        
        # 使用保存的进程树信息（确保和显示时的顺序一致）
        if not self.current_tree_info:
            print("❌ 错误: 没有可用的进程树信息，请重新运行程序")
            return
        
        tree_info = self.current_tree_info
        
        # 验证编号范围
        max_tree_count = len(tree_info)
        valid_numbers = []
        for num in tree_numbers:
            if 1 <= num <= max_tree_count:
                valid_numbers.append(num)
            else:
                print(f"⚠️ 警告: 编号 {num} 超出范围 (1-{max_tree_count})")
        
        if not valid_numbers:
            print("🚫 没有有效的编号，操作已取消。")
            return
        
        # 获取对应的根进程PID（使用排序后的tree_info）
        target_pids = []
        selected_trees = []
        for num in valid_numbers:
            root, pmem, vsz, rss, gpu_memory, python_count, python_pids = tree_info[num - 1]  # 编号从1开始，索引从0开始
            target_pids.append(root.pid)
            selected_trees.append((num, root))
        
        print(f"\n📊 选中的进程树: {' '.join(map(str, valid_numbers))}")
        print(f"📊 对应的根进程PID: {' '.join(map(str, target_pids))}")
        
        # 显示选中的进程树详细信息
        for num, root in selected_trees:
            print(f"   - 进程树 {num}: PID {root.pid} ({'Python' if root.is_python else '其他'})")
        
        # 询问是否只终止Python进程还是所有进程
        print("\n🤔 请选择终止范围:")
        print("1. 只终止Python进程 (推荐)")
        print("2. 终止所有进程 (包括Python和非Python进程)")
        
        try:
            kill_choice = input("请选择 (1/2): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n🚫 操作已取消，未终止任何进程。")
            return
        
        kill_all_processes = (kill_choice == '2')
        
        # 获取要终止的所有PID
        pids_to_kill = self.get_pids_from_subtrees(target_pids, kill_all_processes)
        
        if not pids_to_kill:
            print("🚫 没有找到要终止的进程，操作已取消。")
            return
        
        # 显示将要终止的进程信息
        print(f"\n📝 基于进程树编号 {' '.join(map(str, valid_numbers))}，即将终止以下进程:")
        
        python_count = 0
        other_count = 0
        
        for pid in pids_to_kill:
            process = self.processes.get(pid)
            if process:
                if process.is_python:
                    python_count += 1
                    mark = "🐍"
                else:
                    other_count += 1
                    mark = "📁"
                
                gpu_info = f" GPU {self.format_memory(process.gpu_memory * 1024):<8}" if process.gpu_memory > 0 else " GPU 0MB      "
                print(f"  {mark} PID {pid:<8} RSS {self.format_memory(process.rss):<8}{gpu_info} "
                      f"运行时间 {self.format_uptime(process.etime):<12}")
        
        print(f"\n📊 统计: Python进程 {python_count}个, 其他进程 {other_count}个, 总计 {len(pids_to_kill)}个")
        
        self._confirm_and_kill(pids_to_kill)
    
    def parse_tree_numbers(self, input_str: str) -> List[int]:
        """解析进程树编号输入，支持范围格式"""
        numbers = []
        parts = input_str.split()
        
        for part in parts:
            if '-' in part:
                # 处理范围格式，如 "1-5"
                try:
                    start_str, end_str = part.split('-', 1)
                    start = int(start_str.strip())
                    end = int(end_str.strip())
                    
                    if start > end:
                        raise ValueError(f"范围格式错误: {part} (起始值不能大于结束值)")
                    
                    numbers.extend(range(start, end + 1))
                except ValueError as e:
                    if "范围格式错误" in str(e):
                        raise e
                    else:
                        raise ValueError(f"范围格式错误: {part} (请使用如 '1-5' 的格式)")
            else:
                # 处理单个数字
                try:
                    numbers.append(int(part.strip()))
                except ValueError:
                    raise ValueError(f"无效的编号: {part}")
        
        # 去重并排序
        return sorted(list(set(numbers)))
    
    def kill_by_pids(self):
        """按指定PID节点终止进程"""
        print("\n🎯 请输入要终止的进程PID (多个PID用空格分隔，回车取消):")
        print("💡 程序会终止这些进程及其所有子进程")
        print("💡 支持格式: 单个PID(1234)、多个PID(1234 5678 9012)")
        print("💡 提示: 要终止整个进程树，请输入其根节点的PID")
        
        try:
            pid_input = input("输入PID: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n🚫 操作已取消，未终止任何进程。")
            return
        
        if not pid_input:
            print("🚫 操作已取消，未终止任何进程。")
            return
        
        # 解析PID列表
        try:
            target_pids = []
            for pid_str in pid_input.split():
                pid = int(pid_str.strip())
                target_pids.append(pid)
        except ValueError:
            print("⚠️ 错误: 请输入有效的PID数字")
            return
        
        if not target_pids:
            print("🚫 没有输入有效的PID，操作已取消。")
            return
        
        # 验证PID是否存在
        valid_pids = []
        for pid in target_pids:
            if pid in self.processes:
                valid_pids.append(pid)
            else:
                print(f"⚠️ 警告: PID {pid} 不在当前进程列表中")
        
        if not valid_pids:
            print("🚫 没有有效的PID，操作已取消。")
            return
        
        print(f"\n📊 分析PID节点: {' '.join(map(str, valid_pids))}")
        
        # 询问是否只终止Python进程还是所有进程
        print("\n🤔 请选择终止范围:")
        print("1. 只终止Python进程 (推荐)")
        print("2. 终止所有进程 (包括Python和非Python进程)")
        
        try:
            kill_choice = input("请选择 (1/2): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n🚫 操作已取消，未终止任何进程。")
            return
        
        kill_all_processes = (kill_choice == '2')
        
        # 获取要终止的所有PID
        pids_to_kill = self.get_pids_from_subtrees(valid_pids, kill_all_processes)
        
        if not pids_to_kill:
            print("🚫 没有找到要终止的进程，操作已取消。")
            return
        
        # 显示将要终止的进程信息
        print(f"\n📝 基于节点 {' '.join(map(str, valid_pids))}，即将终止以下进程:")
        
        process_type = "所有进程" if kill_all_processes else "Python进程"
        python_count = 0
        other_count = 0
        
        for pid in pids_to_kill:
            process = self.processes.get(pid)
            if process:
                if process.is_python:
                    python_count += 1
                    mark = "🐍"
                else:
                    other_count += 1
                    mark = "📁"
                
                gpu_info = f" GPU {self.format_memory(process.gpu_memory * 1024):<8}" if process.gpu_memory > 0 else " GPU 0MB      "
                print(f"  {mark} PID {pid:<8} RSS {self.format_memory(process.rss):<8}{gpu_info} "
                      f"运行时间 {self.format_uptime(process.etime):<12}")
        
        print(f"\n📊 统计: Python进程 {python_count}个, 其他进程 {other_count}个, 总计 {len(pids_to_kill)}个")
        
        self._confirm_and_kill(pids_to_kill)
    
    def _confirm_and_kill(self, pids_to_kill: List[int]):
        """确认并终止进程"""
        try:
            confirm = input("\n❓ 确认终止？[y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n🚫 操作已取消，未终止任何进程。")
            return
        
        if confirm in ('y', 'yes'):
            try:
                success_count = 0
                error_count = 0
                
                for pid in pids_to_kill:
                    try:
                        os.kill(pid, 15)  # SIGTERM
                        success_count += 1
                    except ProcessLookupError:
                        print(f"⚠️ PID {pid} 进程已不存在")
                        error_count += 1
                    except PermissionError:
                        print(f"⚠️ 没有权限终止PID {pid}")
                        error_count += 1
                    except Exception as e:
                        print(f"⚠️ 终止PID {pid} 时出错: {e}")
                        error_count += 1
                
                if success_count > 0:
                    print(f"✅ 成功发送终止信号到 {success_count} 个进程")
                if error_count > 0:
                    print(f"⚠️ {error_count} 个进程终止失败")
                    
            except Exception as e:
                print(f"⚠️ 终止过程中出现错误: {e}")
        else:
            print("🚫 操作已取消，未终止任何进程。")
    
    def run(self):
        """主运行函数"""
        if not self.collect_process_info():
            return
        
        tree_info = self.display_process_trees()
        if tree_info:
            self.interactive_kill()


def main():
    """主函数"""
    try:
        print("🐍 Python进程管理器")
        print("=" * 50)
        print("请选择查看范围:")
        print("1. 只查看当前用户的Python进程 (推荐)")
        print("2. 查看所有用户的Python进程 (需要相应权限)")
        
        try:
            choice = input("请选择 (1/2): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n🚫 操作已取消")
            return
        
        if choice == '1':
            all_users = False
        elif choice == '2':
            all_users = True
            print("⚠️ 注意: 查看所有用户进程可能需要特殊权限")
        else:
            print("❌ 无效选择，使用默认设置（仅当前用户）")
            all_users = False
        
        manager = PythonProcessManager(all_users=all_users)
        manager.run()
    except KeyboardInterrupt:
        print("\n\n🚫 用户中断操作")
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

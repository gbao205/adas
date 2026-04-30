"""
Performance Benchmark - Đo lường hiệu năng real-time
Hỗ trợ:
- FPS (Frames Per Second)
- Latency (ms per frame)
- Memory Usage
- GPU Utilization
"""

import time
import psutil
import numpy as np
from typing import Dict, List, Optional
from collections import deque


class PerformanceBenchmark:
    """
    Đo lường và theo dõi hiệu năng hệ thống real-time.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: Số frame để tính moving average
        """
        self.window_size = window_size
        
        # Lưu trữ timing
        self.frame_times = deque(maxlen=window_size)
        self.inference_times = deque(maxlen=window_size)
        self.preprocessing_times = deque(maxlen=window_size)
        self.postprocessing_times = deque(maxlen=window_size)
        
        # Counters
        self.total_frames = 0
        self.start_time = None
        self.current_frame_start = None
        
        # Memory tracking
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # GPU tracking (nếu có)
        self.gpu_available = False
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_available = True
                self.gpu_memory_allocated = []
                self.gpu_memory_reserved = []
        except ImportError:
            pass
    
    def start_benchmark(self):
        """Bắt đầu benchmark session."""
        self.start_time = time.time()
        print("[Benchmark] Bắt đầu đo lường hiệu năng...")
    
    def start_frame(self):
        """Đánh dấu bắt đầu xử lý một frame."""
        self.current_frame_start = time.time()
    
    def end_frame(self):
        """Đánh dấu kết thúc xử lý một frame."""
        if self.current_frame_start is None:
            return
        
        frame_time = (time.time() - self.current_frame_start) * 1000  # ms
        self.frame_times.append(frame_time)
        self.total_frames += 1
        self.current_frame_start = None
    
    def record_inference_time(self, time_ms: float):
        """Ghi lại thời gian inference."""
        self.inference_times.append(time_ms)
    
    def record_preprocessing_time(self, time_ms: float):
        """Ghi lại thời gian preprocessing."""
        self.preprocessing_times.append(time_ms)
    
    def record_postprocessing_time(self, time_ms: float):
        """Ghi lại thời gian postprocessing."""
        self.postprocessing_times.append(time_ms)
    
    def record_gpu_memory(self):
        """Ghi lại GPU memory usage (nếu có)."""
        if not self.gpu_available:
            return
        
        try:
            import torch
            allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
            self.gpu_memory_allocated.append(allocated)
            self.gpu_memory_reserved.append(reserved)
        except Exception:
            pass
    
    def get_current_fps(self) -> float:
        """Tính FPS hiện tại (moving average)."""
        if not self.frame_times:
            return 0.0
        
        avg_frame_time = np.mean(self.frame_times) / 1000.0  # Convert to seconds
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def get_average_fps(self) -> float:
        """Tính FPS trung bình từ đầu session."""
        if self.start_time is None or self.total_frames == 0:
            return 0.0
        
        elapsed = time.time() - self.start_time
        return self.total_frames / elapsed if elapsed > 0 else 0.0
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Trả về thống kê latency (ms)."""
        if not self.frame_times:
            return {
                'mean': 0.0,
                'min': 0.0,
                'max': 0.0,
                'std': 0.0,
                'p50': 0.0,
                'p95': 0.0,
                'p99': 0.0
            }
        
        times = np.array(self.frame_times)
        return {
            'mean': np.mean(times),
            'min': np.min(times),
            'max': np.max(times),
            'std': np.std(times),
            'p50': np.percentile(times, 50),
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99)
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Trả về thông tin memory usage."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - self.initial_memory
        
        result = {
            'current_mb': current_memory,
            'initial_mb': self.initial_memory,
            'increase_mb': memory_increase,
            'percent': self.process.memory_percent()
        }
        
        # Thêm GPU memory nếu có
        if self.gpu_available and self.gpu_memory_allocated:
            result['gpu_allocated_mb'] = self.gpu_memory_allocated[-1]
            result['gpu_reserved_mb'] = self.gpu_memory_reserved[-1]
        
        return result
    
    def get_breakdown(self) -> Dict[str, float]:
        """Trả về breakdown thời gian xử lý."""
        result = {}
        
        if self.preprocessing_times:
            result['preprocessing_ms'] = np.mean(self.preprocessing_times)
        
        if self.inference_times:
            result['inference_ms'] = np.mean(self.inference_times)
        
        if self.postprocessing_times:
            result['postprocessing_ms'] = np.mean(self.postprocessing_times)
        
        if self.frame_times:
            result['total_ms'] = np.mean(self.frame_times)
        
        return result
    
    def get_summary(self) -> Dict[str, any]:
        """
        Trả về tổng hợp toàn bộ metrics.
        """
        summary = {
            'total_frames': self.total_frames,
            'current_fps': self.get_current_fps(),
            'average_fps': self.get_average_fps(),
            'latency': self.get_latency_stats(),
            'memory': self.get_memory_usage(),
            'breakdown': self.get_breakdown()
        }
        
        return summary
    
    def print_summary(self):
        """In ra báo cáo tổng hợp."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*60)
        
        print(f"\n📊 THROUGHPUT:")
        print(f"  Total Frames: {summary['total_frames']}")
        print(f"  Current FPS: {summary['current_fps']:.2f}")
        print(f"  Average FPS: {summary['average_fps']:.2f}")
        
        print(f"\n⏱️  LATENCY (ms):")
        latency = summary['latency']
        print(f"  Mean: {latency['mean']:.2f}")
        print(f"  Min: {latency['min']:.2f}")
        print(f"  Max: {latency['max']:.2f}")
        print(f"  Std: {latency['std']:.2f}")
        print(f"  P95: {latency['p95']:.2f}")
        print(f"  P99: {latency['p99']:.2f}")
        
        if summary['breakdown']:
            print(f"\n🔧 BREAKDOWN:")
            breakdown = summary['breakdown']
            if 'preprocessing_ms' in breakdown:
                print(f"  Preprocessing: {breakdown['preprocessing_ms']:.2f} ms")
            if 'inference_ms' in breakdown:
                print(f"  Inference: {breakdown['inference_ms']:.2f} ms")
            if 'postprocessing_ms' in breakdown:
                print(f"  Postprocessing: {breakdown['postprocessing_ms']:.2f} ms")
            if 'total_ms' in breakdown:
                print(f"  Total: {breakdown['total_ms']:.2f} ms")
        
        print(f"\n💾 MEMORY:")
        memory = summary['memory']
        print(f"  Current: {memory['current_mb']:.2f} MB")
        print(f"  Increase: {memory['increase_mb']:.2f} MB")
        print(f"  Percent: {memory['percent']:.2f}%")
        
        if 'gpu_allocated_mb' in memory:
            print(f"  GPU Allocated: {memory['gpu_allocated_mb']:.2f} MB")
            print(f"  GPU Reserved: {memory['gpu_reserved_mb']:.2f} MB")
        
        print("\n" + "="*60)
    
    def is_realtime(self, target_fps: float = 30.0) -> bool:
        """
        Kiểm tra xem hệ thống có đạt real-time không.
        
        Args:
            target_fps: FPS mục tiêu (mặc định 30)
        
        Returns:
            True nếu đạt target FPS
        """
        current_fps = self.get_current_fps()
        return current_fps >= target_fps
    
    def get_realtime_status(self, target_fps: float = 30.0) -> str:
        """
        Trả về trạng thái real-time dạng text.
        """
        current_fps = self.get_current_fps()
        
        if current_fps >= target_fps:
            return f"✅ REAL-TIME ({current_fps:.1f} FPS >= {target_fps} FPS)"
        else:
            deficit = target_fps - current_fps
            return f"❌ NOT REAL-TIME ({current_fps:.1f} FPS, thiếu {deficit:.1f} FPS)"

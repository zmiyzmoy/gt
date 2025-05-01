"""
Memory tracking utilities for monitoring Ray/RLlib memory usage.

This module provides classes to track memory usage during RL training,
helping to identify memory leaks and resource constraints.
"""

import os
import time
import logging
import psutil
import json
import threading
from pathlib import Path
import gc
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False

logger = logging.getLogger(__name__)

class MemoryTracker:
    """
    Track memory usage over time, with optional Ray integration.
    
    This class monitors system memory usage and can optionally query Ray
    for additional memory information from the cluster.
    """
    
    def __init__(self, log_dir='./memory_logs', interval=30, 
                 ray_monitoring=True, plot_results=True,
                 max_points=1000):
        """
        Initialize memory tracker.
        
        Args:
            log_dir: Directory to save memory logs
            interval: Monitoring interval in seconds
            ray_monitoring: Whether to monitor Ray cluster memory
            plot_results: Whether to generate memory usage plots
            max_points: Maximum number of data points to keep in memory
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.interval = interval
        self.ray_monitoring = ray_monitoring and HAS_RAY
        self.plot_results = plot_results
        self.max_points = max_points
        
        self.process = psutil.Process(os.getpid())
        self.stop_event = threading.Event()
        self.tracking_thread = None
        
        # Data storage
        self.timestamps = []
        self.system_memory = []
        self.process_memory = []
        self.ray_memory = []
        
        # Filename setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"memory_log_{timestamp}.json"
        self.plot_file = self.log_dir / f"memory_plot_{timestamp}.png"
        
        logger.info(f"Memory tracker initialized. Logs will be saved to {self.log_dir}")
    
    def start(self):
        """Start memory tracking in a background thread."""
        if self.tracking_thread is not None and self.tracking_thread.is_alive():
            logger.warning("Memory tracker is already running")
            return
            
        self.stop_event.clear()
        self.tracking_thread = threading.Thread(
            target=self._tracking_loop,
            daemon=True
        )
        self.tracking_thread.start()
        logger.info(f"Memory tracking started with {self.interval}s interval")
        
    def stop(self):
        """Stop memory tracking and save results."""
        if self.tracking_thread is None or not self.tracking_thread.is_alive():
            logger.warning("Memory tracker is not running")
            return
            
        self.stop_event.set()
        self.tracking_thread.join(timeout=5)
        
        if self.tracking_thread.is_alive():
            logger.warning("Memory tracking thread did not stop cleanly")
        else:
            logger.info("Memory tracking stopped")
            
        self.save_results()
        
        if self.plot_results and len(self.timestamps) > 1:
            self.generate_plot()
    
    def _tracking_loop(self):
        """Main memory tracking loop."""
        while not self.stop_event.is_set():
            try:
                self._collect_memory_stats()
            except Exception as e:
                logger.error(f"Error collecting memory stats: {e}")
            
            # Sleep for the interval or until stopped
            self.stop_event.wait(self.interval)
    
    def _collect_memory_stats(self):
        """Collect memory statistics."""
        timestamp = time.time()
        
        # System memory
        system_mem = psutil.virtual_memory()
        system_mem_used_gb = system_mem.used / (1024**3)
        
        # Process memory
        process_mem_gb = self.process.memory_info().rss / (1024**3)
        
        # Ray memory (if available)
        ray_mem_gb = 0
        if self.ray_monitoring and ray.is_initialized():
            try:
                # This is not an official API and might change
                ray_memory_info = ray.cluster_resources().get("memory", 0)
                ray_mem_gb = ray_memory_info / (1024**3)
            except Exception as e:
                logger.debug(f"Error getting Ray memory info: {e}")
        
        # Store data
        self.timestamps.append(timestamp)
        self.system_memory.append(system_mem_used_gb)
        self.process_memory.append(process_mem_gb)
        self.ray_memory.append(ray_mem_gb)
        
        # Keep only the last max_points
        if len(self.timestamps) > self.max_points:
            self.timestamps = self.timestamps[-self.max_points:]
            self.system_memory = self.system_memory[-self.max_points:]
            self.process_memory = self.process_memory[-self.max_points:]
            self.ray_memory = self.ray_memory[-self.max_points:]
        
        # Log periodically
        if len(self.timestamps) % 10 == 0:
            logger.debug(f"Memory usage - System: {system_mem_used_gb:.2f} GB, "
                         f"Process: {process_mem_gb:.2f} GB, "
                         f"Ray: {ray_mem_gb:.2f} GB")
    
    def save_results(self):
        """Save memory tracking results to a file."""
        if not self.timestamps:
            logger.warning("No memory data to save")
            return
            
        data = {
            "timestamps": self.timestamps,
            "system_memory_gb": self.system_memory,
            "process_memory_gb": self.process_memory,
            "ray_memory_gb": self.ray_memory,
            "start_time": self.timestamps[0],
            "end_time": self.timestamps[-1],
            "duration_seconds": self.timestamps[-1] - self.timestamps[0],
            "samples": len(self.timestamps),
            "max_system_memory_gb": max(self.system_memory),
            "max_process_memory_gb": max(self.process_memory),
            "max_ray_memory_gb": max(self.ray_memory)
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Memory tracking data saved to {self.log_file}")
    
    def generate_plot(self):
        """Generate a plot of memory usage over time."""
        if not self.timestamps:
            logger.warning("No memory data to plot")
            return
            
        try:
            # Convert timestamps to readable time
            relative_times = [(t - self.timestamps[0]) / 60 for t in self.timestamps]  # Minutes
            
            plt.figure(figsize=(12, 6))
            
            # Plot memory usage
            plt.plot(relative_times, self.system_memory, label="System Memory (GB)")
            plt.plot(relative_times, self.process_memory, label="Process Memory (GB)")
            
            if any(x > 0 for x in self.ray_memory):
                plt.plot(relative_times, self.ray_memory, label="Ray Memory (GB)")
            
            plt.xlabel("Time (minutes)")
            plt.ylabel("Memory Usage (GB)")
            plt.title("Poker RL Training Memory Usage")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add annotations for peak memory
            max_sys_idx = np.argmax(self.system_memory)
            max_proc_idx = np.argmax(self.process_memory)
            
            plt.annotate(f"{self.system_memory[max_sys_idx]:.2f} GB", 
                        xy=(relative_times[max_sys_idx], self.system_memory[max_sys_idx]),
                        xytext=(10, 10), textcoords="offset points",
                        arrowprops=dict(arrowstyle="->"))
                        
            plt.annotate(f"{self.process_memory[max_proc_idx]:.2f} GB", 
                        xy=(relative_times[max_proc_idx], self.process_memory[max_proc_idx]),
                        xytext=(10, -10), textcoords="offset points",
                        arrowprops=dict(arrowstyle="->"))
            
            plt.tight_layout()
            plt.savefig(self.plot_file)
            plt.close()
            
            logger.info(f"Memory usage plot saved to {self.plot_file}")
        except Exception as e:
            logger.error(f"Error generating memory plot: {e}")
    
    def get_memory_summary(self):
        """Get a summary of memory usage statistics."""
        if not self.timestamps:
            return "No memory data collected"
            
        duration_minutes = (self.timestamps[-1] - self.timestamps[0]) / 60
        
        return (
            f"Memory tracking summary:\n"
            f"Duration: {duration_minutes:.2f} minutes\n"
            f"Peak system memory: {max(self.system_memory):.2f} GB\n"
            f"Peak process memory: {max(self.process_memory):.2f} GB\n"
            f"Peak Ray memory: {max(self.ray_memory):.2f} GB\n"
            f"Latest system memory: {self.system_memory[-1]:.2f} GB\n"
            f"Latest process memory: {self.process_memory[-1]:.2f} GB\n"
        )
    
    def trigger_garbage_collection(self):
        """Force garbage collection and record memory change."""
        if not self.timestamps:
            return
            
        # Record before GC
        before_gc = self.process.memory_info().rss / (1024**3)
        
        # Force garbage collection
        gc.collect()
        
        # Record after GC
        after_gc = self.process.memory_info().rss / (1024**3)
        
        logger.info(f"Garbage collection released {before_gc - after_gc:.2f} GB of memory")
        
        # Record this point
        self._collect_memory_stats()
        
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

class RayMemoryMonitor:
    """
    A specialized memory monitor for Ray clusters.
    
    This class collects detailed memory information from Ray workers
    and can help identify memory leaks in distributed training.
    """
    
    def __init__(self, log_dir='./ray_memory_logs', interval=60):
        """
        Initialize Ray memory monitor.
        
        Args:
            log_dir: Directory to save memory logs
            interval: Monitoring interval in seconds
        """
        if not HAS_RAY:
            raise ImportError("Ray is required for RayMemoryMonitor")
            
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.interval = interval
        self.stop_event = threading.Event()
        self.monitoring_thread = None
        
        # Data storage
        self.memory_data = []
        
        # Filename setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"ray_memory_{timestamp}.json"
        
        logger.info(f"Ray memory monitor initialized. Logs will be saved to {self.log_dir}")
    
    def start(self):
        """Start Ray memory monitoring."""
        if not ray.is_initialized():
            logger.error("Ray is not initialized. Cannot start monitoring.")
            return False
            
        if self.monitoring_thread is not None and self.monitoring_thread.is_alive():
            logger.warning("Ray memory monitor is already running")
            return True
            
        self.stop_event.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"Ray memory monitoring started with {self.interval}s interval")
        return True
        
    def stop(self):
        """Stop Ray memory monitoring and save results."""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            logger.warning("Ray memory monitor is not running")
            return
            
        self.stop_event.set()
        self.monitoring_thread.join(timeout=5)
        
        if self.monitoring_thread.is_alive():
            logger.warning("Ray monitoring thread did not stop cleanly")
        else:
            logger.info("Ray memory monitoring stopped")
            
        self.save_results()
    
    def _monitoring_loop(self):
        """Main Ray memory monitoring loop."""
        while not self.stop_event.is_set():
            try:
                self._collect_ray_memory_stats()
            except Exception as e:
                logger.error(f"Error collecting Ray memory stats: {e}")
            
            # Sleep for the interval or until stopped
            self.stop_event.wait(self.interval)
    
    def _collect_ray_memory_stats(self):
        """Collect detailed Ray memory statistics."""
        if not ray.is_initialized():
            logger.error("Ray is not initialized")
            return
            
        try:
            timestamp = time.time()
            
            # Get basic cluster info
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()
            
            # Get node information
            nodes_info = ray.nodes()
            
            # Collect memory stats for each node
            nodes_memory = []
            for node in nodes_info:
                node_id = node["NodeID"]
                ip = node["NodeManagerAddress"]
                
                # Get memory metrics
                mem_total = node.get("Resources", {}).get("memory", 0) / (1024**3)
                mem_available = node.get("Resources", {}).get("memory", 0) / (1024**3)
                
                # Custom memory stats for this node
                node_mem = {
                    "node_id": node_id,
                    "ip": ip,
                    "total_memory_gb": mem_total,
                    "available_memory_gb": mem_available,
                    "used_memory_gb": mem_total - mem_available
                }
                nodes_memory.append(node_mem)
            
            # Store the complete memory information
            memory_info = {
                "timestamp": timestamp,
                "total_cluster_memory_gb": cluster_resources.get("memory", 0) / (1024**3),
                "available_cluster_memory_gb": available_resources.get("memory", 0) / (1024**3),
                "node_count": len(nodes_info),
                "nodes": nodes_memory
            }
            
            self.memory_data.append(memory_info)
            
            # Log occasionally
            if len(self.memory_data) % 5 == 0:
                used_memory = memory_info["total_cluster_memory_gb"] - memory_info["available_cluster_memory_gb"]
                logger.info(f"Ray cluster memory: {used_memory:.2f} GB used out of {memory_info['total_cluster_memory_gb']:.2f} GB")
                
        except Exception as e:
            logger.error(f"Error collecting Ray memory stats: {e}")
    
    def save_results(self):
        """Save Ray memory monitoring results to a file."""
        if not self.memory_data:
            logger.warning("No Ray memory data to save")
            return
            
        with open(self.log_file, 'w') as f:
            json.dump({
                "ray_memory_data": self.memory_data,
                "start_time": self.memory_data[0]["timestamp"],
                "end_time": self.memory_data[-1]["timestamp"],
                "duration_seconds": self.memory_data[-1]["timestamp"] - self.memory_data[0]["timestamp"],
                "samples": len(self.memory_data)
            }, f, indent=2)
            
        logger.info(f"Ray memory monitoring data saved to {self.log_file}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

def monitor_training_memory(func):
    """Decorator to monitor memory during training functions."""
    def wrapper(*args, **kwargs):
        # Start memory monitoring
        memory_tracker = MemoryTracker(interval=30)
        memory_tracker.start()
        
        try:
            # Run the training function
            result = func(*args, **kwargs)
            return result
        finally:
            # Stop monitoring and save results
            memory_tracker.stop()
            logger.info(memory_tracker.get_memory_summary())
    
    return wrapper

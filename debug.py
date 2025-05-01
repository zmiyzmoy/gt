#!/usr/bin/env python3
"""
Debugging script for Poker RL System.
This script verifies the environment, dependencies, and system resources
before running the training pipeline.
"""

import os
import sys
import argparse
import logging
import platform
import subprocess
import gc
import traceback
import warnings
import json
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('debug.log')
    ]
)
logger = logging.getLogger("poker_debug")

# Try to import dependencies
logger.info("Checking dependencies...")

def check_dependency(module_name, min_version=None):
    """Check if a dependency is installed and at minimum version."""
    try:
        module = __import__(module_name)
        if hasattr(module, '__version__'):
            version = module.__version__
        elif hasattr(module, 'VERSION'):
            version = module.VERSION
        else:
            version = "Unknown"
        
        if min_version and version != "Unknown":
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                logger.warning(f"✗ {module_name} version {version} is below recommended {min_version}")
                return False, version
        
        logger.info(f"✓ {module_name} version {version}")
        return True, version
    except ImportError:
        logger.error(f"✗ {module_name} is not installed")
        return False, None
    except Exception as e:
        logger.error(f"✗ Error checking {module_name}: {e}")
        return False, None

# Check system information
def check_system_info():
    """Get system information."""
    logger.info("System Information:")
    logger.info(f"OS: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    
    try:
        import psutil
        mem = psutil.virtual_memory()
        logger.info(f"Memory: {mem.total / (1024**3):.2f} GB total, {mem.available / (1024**3):.2f} GB available")
        
        if mem.available < 4 * (1024**3):  # Less than 4GB available
            logger.warning("Low memory available! RLlib training may fail.")
        
        cpu_count = psutil.cpu_count(logical=False)
        logger.info(f"CPU: {cpu_count} physical cores, {psutil.cpu_count()} logical cores")
        
        if platform.system() == "Linux":
            try:
                # Check for GPU on Linux
                with open('/proc/driver/nvidia/version', 'r') as f:
                    nvidia_info = f.read().strip()
                    logger.info(f"NVIDIA driver: {nvidia_info}")
            except:
                logger.info("No NVIDIA driver detected")
                
        # Check disk space
        disk = psutil.disk_usage('/')
        logger.info(f"Disk: {disk.total / (1024**3):.2f} GB total, {disk.free / (1024**3):.2f} GB free")
        
    except ImportError:
        logger.warning("psutil not installed, skipping detailed system info")
    except Exception as e:
        logger.warning(f"Error getting system info: {e}")

# Check GPU availability
def check_gpu():
    """Check GPU availability for PyTorch."""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        logger.info(f"PyTorch CUDA available: {gpu_available}")
        
        if gpu_available:
            gpu_count = torch.cuda.device_count()
            logger.info(f"GPU count: {gpu_count}")
            for i in range(gpu_count):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                
            # Check memory
            for i in range(gpu_count):
                mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                logger.info(f"GPU {i} memory: {mem_total:.2f} GB total, {mem_reserved:.2f} GB reserved, {mem_allocated:.2f} GB allocated")
                
        return gpu_available
    except Exception as e:
        logger.error(f"Error checking GPU: {e}")
        return False

# Check Ray cluster status
def check_ray():
    """Check Ray cluster status."""
    try:
        import ray
        
        if ray.is_initialized():
            logger.info("Ray is already initialized. Shutting down existing cluster...")
            ray.shutdown()
            
        # Initialize Ray with minimal resources to test
        ray.init(num_cpus=1, include_dashboard=False, ignore_reinit_error=True, log_to_driver=True)
        
        cluster_info = ray.cluster_resources()
        logger.info(f"Ray initialized with resources: {cluster_info}")
        
        # Test creating an actor
        @ray.remote
        class TestActor:
            def ping(self):
                import platform
                return f"pong from {platform.node()}"
        
        test_actor = TestActor.remote()
        result = ray.get(test_actor.ping.remote())
        logger.info(f"Ray actor test: {result}")
        
        # Clean up
        ray.kill(test_actor)
        ray.shutdown()
        logger.info("Ray test completed and shut down successfully")
        return True
    except Exception as e:
        logger.error(f"Ray initialization error: {e}")
        traceback.print_exc()
        return False

# Test OpenSpiel environment
def test_open_spiel():
    """Test OpenSpiel installation and game creation."""
    try:
        import pyspiel
        from open_spiel.python import rl_environment
        
        # List available games
        games = pyspiel.registered_games()
        logger.info(f"OpenSpiel registered games count: {len(games)}")
        
        # Check for universal_poker
        poker_game = next((g for g in games if g.short_name == "universal_poker"), None)
        if poker_game:
            logger.info(f"Found universal_poker game: {poker_game.short_name}")
            logger.info(f"Default parameters: {poker_game.default_loadable}")
        else:
            logger.warning("universal_poker game not found!")
            return False
            
        # Try to create a simple game to test
        try:
            test_game = pyspiel.load_game("universal_poker(betting=nolimit,numPlayers=2,numRounds=4,blind=1 2,stack=1000 1000)")
            logger.info(f"Test game created successfully: {test_game}")
            
            # Test RL environment
            env = rl_environment.Environment(test_game)
            time_step = env.reset()
            logger.info(f"RL environment created, initial state: {time_step.observations['current_player']}")
            return True
        except Exception as e:
            logger.error(f"Error creating test game: {e}")
            return False
    except ImportError:
        logger.error("OpenSpiel not installed correctly")
        return False
    except Exception as e:
        logger.error(f"OpenSpiel test error: {e}")
        return False

# Test our custom environment
def test_custom_environment():
    """Test our custom poker environment."""
    try:
        # Try to import from current directory
        sys.path.append('.')
        from config import PokerConfig
        from environment import PokerEnv
        
        logger.info("Testing custom poker environment...")
        
        # Create minimal config
        config = PokerConfig()
        config.num_players = 2  # Use smaller game for testing
        
        # Create environment
        env_config = {"config": config}
        env = PokerEnv(env_config)
        
        logger.info(f"Environment created successfully, action space: {env.action_space}")
        logger.info(f"Observation space: {env.observation_space}")
        
        # Try a reset
        obs = env.reset()
        logger.info(f"Environment reset, observation keys: {obs.keys() if isinstance(obs, dict) else 'not a dict'}")
        
        return True
    except Exception as e:
        logger.error(f"Custom environment test error: {e}")
        traceback.print_exc()
        return False

# Check model definition
def test_model():
    """Test the model definition."""
    try:
        import torch
        import numpy as np
        import gymnasium as gym
        from ray.rllib.models.modelv2 import ModelV2
        
        # Try to import our model
        from models import AdvancedPokerModel
        
        logger.info("Testing model definition...")
        
        # Create mock observation space
        obs_feature_size = 100  # Just a test value
        obs_space = gym.spaces.Dict({
            "obs": gym.spaces.Box(low=-1, high=1, shape=(obs_feature_size,), dtype=np.float32),
            "action_mask": gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)  # Assuming 3 actions
        })
        
        # Create mock action space
        action_space = gym.spaces.Discrete(3)
        
        # Try to instantiate the model
        model = AdvancedPokerModel(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=3,
            model_config={"fcnet_hiddens": [64, 64]},
            name="test_model"
        )
        
        logger.info(f"Model created successfully: {model}")
        
        # Test forward pass with dummy input
        dummy_input = {
            "obs": {
                "obs": torch.zeros((1, obs_feature_size), dtype=torch.float32),
                "action_mask": torch.ones((1, 3), dtype=torch.float32)
            }
        }
        
        outputs, _ = model(dummy_input, [], None)
        logger.info(f"Model forward pass successful, output shape: {outputs.shape}")
        
        value = model.value_function()
        logger.info(f"Value function output shape: {value.shape}")
        
        return True
    except Exception as e:
        logger.error(f"Model test error: {e}")
        traceback.print_exc()
        return False

# Check for memory leaks
def check_for_memory_leaks():
    """Basic check for memory leaks."""
    try:
        import psutil
        import time
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)
        logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
        
        # Force garbage collection
        gc.collect()
        
        # Check memory again
        time.sleep(1)
        after_gc_memory = process.memory_info().rss / (1024 * 1024)
        logger.info(f"Memory after GC: {after_gc_memory:.2f} MB")
        
        # Create and delete some objects to test garbage collection
        for _ in range(5):
            large_list = [0] * 1000000
            del large_list
            gc.collect()
            
        # Check memory again
        final_memory = process.memory_info().rss / (1024 * 1024)
        logger.info(f"Final memory usage: {final_memory:.2f} MB")
        
        if final_memory > initial_memory * 1.5:
            logger.warning(f"Possible memory leak detected: {final_memory - initial_memory:.2f} MB increase")
        else:
            logger.info("No obvious memory leak detected")
    except Exception as e:
        logger.error(f"Error checking for memory leaks: {e}")

# Test configuration loading
def test_config():
    """Test configuration loading."""
    try:
        from config import PokerConfig, TrainingConfig
        
        poker_config = PokerConfig()
        training_config = TrainingConfig()
        
        logger.info(f"PokerConfig loaded: {poker_config.game_name}, {poker_config.num_players} players")
        logger.info(f"TrainingConfig loaded: batch size {training_config.train_batch_size}, lr {training_config.lr}")
        
        # Test potential memory issues
        logger.info("Checking if config is too resource-intensive...")
        
        if training_config.train_batch_size > 32768:
            logger.warning(f"Large batch size: {training_config.train_batch_size}. Consider reducing it.")
            
        if training_config.num_workers > 8:
            logger.warning(f"High number of workers: {training_config.num_workers}. Consider reducing for VPS.")
            
        return True
    except Exception as e:
        logger.error(f"Config test error: {e}")
        return False

def generate_resource_recommendations():
    """Generate resource recommendations based on system info."""
    try:
        import psutil
        from config import TrainingConfig
        
        mem = psutil.virtual_memory()
        cpu_count = psutil.cpu_count(logical=False)
        
        # Load default config
        default_config = TrainingConfig()
        
        # Calculate recommended settings based on available resources
        # Rule of thumb: leave 1 CPU for the OS and trainer
        recommended_workers = max(1, cpu_count - 1)
        
        # Estimate memory per worker (very rough approximation)
        available_mem_gb = mem.available / (1024**3)
        buffer_gb = 2.0  # Reserve memory for OS and other processes
        usable_mem_gb = max(0.5, available_mem_gb - buffer_gb)
        
        # Memory heuristic: ~1GB per worker for poker + overhead
        max_workers_by_mem = max(1, int(usable_mem_gb / 1.5))
        
        # Take the minimum of CPU-based and memory-based recommendations
        final_recommended_workers = min(recommended_workers, max_workers_by_mem)
        
        # Batch size heuristic: ~0.5GB per 10000 batch size
        rec_batch_size = min(default_config.train_batch_size, 
                            int((usable_mem_gb * 0.6) * 20000))
        # Round to nearest 1024
        rec_batch_size = (rec_batch_size // 1024) * 1024
        if rec_batch_size < 1024:
            rec_batch_size = 1024
            
        # SGD minibatch size should be smaller than batch size
        rec_sgd_batch = min(default_config.sgd_minibatch_size, rec_batch_size // 4)
        rec_sgd_batch = max(128, (rec_sgd_batch // 128) * 128)  # Round to nearest 128
        
        recommendations = {
            "num_workers": final_recommended_workers,
            "train_batch_size": rec_batch_size,
            "sgd_minibatch_size": rec_sgd_batch,
            "num_gpus": 1 if check_gpu() else 0,
            "num_envs_per_worker": 1,  # Conservative default
            "rollout_fragment_length": min(200, default_config.rollout_fragment_length),
        }
        
        logger.info("Resource recommendations for your system:")
        for key, value in recommendations.items():
            logger.info(f"  {key}: {value}")
            
        # Save recommendations to a file
        with open('poker_recommendations.json', 'w') as f:
            json.dump(recommendations, f, indent=2)
        logger.info("Recommendations saved to poker_recommendations.json")
        
        return recommendations
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description="Debug Poker RL System")
    parser.add_argument('--full', action='store_true', help='Run full diagnostic suite')
    parser.add_argument('--gpu', action='store_true', help='Check GPU availability')
    parser.add_argument('--deps', action='store_true', help='Check dependencies')
    parser.add_argument('--ray', action='store_true', help='Test Ray initialization')
    parser.add_argument('--env', action='store_true', help='Test environment')
    parser.add_argument('--model', action='store_true', help='Test model definition')
    parser.add_argument('--optimize', action='store_true', help='Generate optimized settings')
    args = parser.parse_args()
    
    logger.info("=" * 50)
    logger.info("Poker RL System Diagnostics")
    logger.info("=" * 50)
    
    run_all = args.full or not any([args.gpu, args.deps, args.ray, args.env, args.model, args.optimize])
    
    if run_all or args.deps:
        logger.info("\n--- Checking Dependencies ---")
        dependencies = [
            ("numpy", "1.20.0"),
            ("ray", "2.10.0"),
            ("torch", "2.0.1"),
            ("gymnasium", "0.29.1"),
            ("pyspiel", None),
            ("open_spiel", "1.3"),
            ("wandb", None)
        ]
        
        all_deps_ok = True
        for dep, min_ver in dependencies:
            ok, _ = check_dependency(dep, min_ver)
            all_deps_ok = all_deps_ok and ok
        
        if not all_deps_ok:
            logger.warning("Not all dependencies are properly installed!")
    
    if run_all or args.gpu:
        logger.info("\n--- Checking System Info ---")
        check_system_info()
        
        logger.info("\n--- Checking GPU ---")
        check_gpu()
    
    if run_all or args.ray:
        logger.info("\n--- Testing Ray ---")
        check_ray()
    
    if run_all or args.env:
        logger.info("\n--- Testing OpenSpiel ---")
        test_open_spiel()
        
        logger.info("\n--- Testing Custom Environment ---")
        test_custom_environment()
    
    if run_all or args.model:
        logger.info("\n--- Testing Model ---")
        test_model()
        
        logger.info("\n--- Testing Config ---")
        test_config()
        
        logger.info("\n--- Checking for Memory Leaks ---")
        check_for_memory_leaks()
    
    if run_all or args.optimize:
        logger.info("\n--- Generating Resource Recommendations ---")
        generate_resource_recommendations()
    
    logger.info("\n--- Diagnostics Summary ---")
    logger.info("Check the debug.log file for detailed information")
    logger.info("If issues persist, try running with reduced resources using the recommendations")

if __name__ == "__main__":
    main()

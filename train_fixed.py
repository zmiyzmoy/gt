# train_fixed.py
"""
Fixed version of the Poker training script with memory optimizations
and improved error handling for reliable training on VPS environments.
"""
import argparse
import os
import logging
import gc
import time
import json
from pathlib import Path
import numpy as np
import psutil

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog

# Import configuration and environment
from config import PokerConfig, TrainingConfig
from environment_fixed import PokerEnv
from models import AdvancedPokerModel

# Import memory tracking utilities
try:
    from utils.memory_tracker import MemoryTracker, monitor_training_memory
except ImportError:
    # Fallback if utils not available
    def monitor_training_memory(func):
        return func
    MemoryTracker = None

# Set up logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('train.log')
    ]
)
logger = logging.getLogger("poker_training")

def create_env(env_config):
    """Factory function for creating environment instances."""
    return PokerEnv(env_config)

def get_available_resources():
    """Get information about available system resources."""
    try:
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count()
        
        resources = {
            "memory_total_gb": memory.total / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "cpu_count": cpu_count,
            "memory_percent": memory.percent
        }
        
        logger.info(f"Available resources: {resources}")
        return resources
    except Exception as e:
        logger.error(f"Error getting system resources: {e}")
        return {
            "memory_available_gb": 4.0,  # Conservative default
            "cpu_count": 2             # Conservative default
        }

def optimize_config_for_resources(train_cfg):
    """Adjust training configuration based on available resources."""
    resources = get_available_resources()
    
    # Only modify if we're resource constrained
    if resources["memory_available_gb"] < 8.0 or resources["cpu_count"] < 4:
        logger.warning("Limited resources detected, optimizing configuration...")
        
        # Reduce workers based on CPU count (leave 1 for system)
        max_workers = max(1, resources["cpu_count"] - 1)
        if train_cfg.num_workers > max_workers:
            logger.info(f"Reducing workers from {train_cfg.num_workers} to {max_workers}")
            train_cfg.num_workers = max_workers
        
        # Reduce batch size if memory is limited
        if resources["memory_available_gb"] < 6.0 and train_cfg.train_batch_size > 4096:
            new_batch_size = 4096
            logger.info(f"Reducing train_batch_size from {train_cfg.train_batch_size} to {new_batch_size}")
            train_cfg.train_batch_size = new_batch_size
            
            # Adjust SGD minibatch size too
            new_sgd_size = min(train_cfg.sgd_minibatch_size, new_batch_size // 4)
            logger.info(f"Reducing sgd_minibatch_size from {train_cfg.sgd_minibatch_size} to {new_sgd_size}")
            train_cfg.sgd_minibatch_size = new_sgd_size
        
        # Use lower precision if very memory constrained
        if resources["memory_available_gb"] < 4.0:
            logger.info("Enabling mixed precision training to save memory")
            os.environ["MIXED_PRECISION"] = "1"  # Will be used to set precision in model
    
    return train_cfg

def setup_ray(log_level="INFO"):
    """Initialize Ray with proper error handling."""
    if ray.is_initialized():
        logger.info("Ray is already initialized. Shutting down and reinitializing...")
        ray.shutdown()
    
    try:
        # Start with minimal resources to avoid overcommitting
        ray.init(
            logging_level=log_level,
            log_to_driver=True,
            # Configure object store to prevent out-of-memory issues
            object_store_memory=None,  # Let Ray auto-configure
            _memory=None,              # Let Ray auto-configure
            # Add useful system metrics
            include_dashboard=False
        )
        
        logger.info("Ray initialized successfully")
        logger.info(f"Ray cluster resources: {ray.cluster_resources()}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Ray: {e}", exc_info=True)
        
        # Try with more conservative settings
        try:
            # Calculate conservative memory limit (50% of available)
            memory = psutil.virtual_memory()
            conservative_memory = int(memory.available * 0.5)
            
            logger.info(f"Trying conservative Ray initialization with {conservative_memory/(1024**3):.2f} GB memory limit")
            
            ray.init(
                logging_level=log_level,
                log_to_driver=True,
                object_store_memory=int(conservative_memory * 0.5),
                _memory=int(conservative_memory * 0.5),
                include_dashboard=False
            )
            
            logger.info("Ray initialized with conservative settings")
            logger.info(f"Ray cluster resources: {ray.cluster_resources()}")
            return True
        except Exception as e2:
            logger.error(f"Conservative Ray initialization failed: {e2}", exc_info=True)
            return False

def get_best_checkpoint(analysis, metric="episode_reward_mean"):
    """Safely retrieve the best checkpoint from analysis."""
    try:
        if not analysis.trials:
            logger.warning("No trials completed, cannot get best checkpoint")
            return None
            
        # Try evaluation metric first
        eval_metric = "evaluation/episode_reward_mean"
        best_trial = analysis.get_best_trial(eval_metric, mode="max", scope="last")
        
        if not best_trial:
            # Fall back to training metric
            best_trial = analysis.get_best_trial(metric, mode="max", scope="last")
            
        if not best_trial:
            logger.warning(f"Could not find best trial using metrics {eval_metric} or {metric}")
            return None
            
        logger.info(f"Best trial: {best_trial.trial_id}")
        
        # Get best checkpoint for this trial
        checkpoint_result = analysis.get_best_checkpoint(
            trial=best_trial,
            metric=metric,
            mode="max"
        )
        
        if checkpoint_result:
            checkpoint_path = getattr(checkpoint_result, 'path', str(checkpoint_result))
            logger.info(f"Best checkpoint path: {checkpoint_path}")
            return checkpoint_path
        else:
            logger.warning(f"No checkpoint for best trial {best_trial.trial_id}")
            return None
    except Exception as e:
        logger.error(f"Error getting best checkpoint: {e}")
        return None

@monitor_training_memory
def train_poker(poker_cfg: PokerConfig, train_cfg: TrainingConfig):
    """Main training function."""
    logger.info("=== Starting Poker RL Training ===")
    
    # Check & optimize resources
    train_cfg = optimize_config_for_resources(train_cfg)
    
    # Register custom model
    try:
        ModelCatalog.register_custom_model("AdvancedPokerModel", AdvancedPokerModel)
        logger.info(f"Custom model '{poker_cfg.model_config['custom_model']}' registered successfully")
    except Exception as e:
        logger.error(f"Failed to register custom model: {e}", exc_info=True)
        raise
    
    # Initialize Ray
    if not setup_ray(train_cfg.log_level):
        logger.error("Ray initialization failed, cannot continue")
        return None
    
    # Configure environment
    env_creator_config = {"config": poker_cfg, "dtype": np.float32}
    
    # Configure algorithm with memory efficiency optimizations
    algo_config = (
        PPOConfig()
        .environment(env=PokerEnv, env_config=env_creator_config)
        .framework("torch")
        .resources(
            num_gpus=train_cfg.num_gpus,
            num_cpus_per_worker=train_cfg.num_cpus_per_worker,
            num_gpus_per_worker=train_cfg.num_gpus_per_worker,
        )
        .rollouts(
            num_rollout_workers=train_cfg.num_workers,
            num_envs_per_worker=train_cfg.num_envs_per_worker,
            rollout_fragment_length=train_cfg.rollout_fragment_length,
            batch_mode=train_cfg.batch_mode,
        )
        .training(
            gamma=train_cfg.gamma,
            lr=train_cfg.lr,
            lambda_=train_cfg.lambda_,
            clip_param=train_cfg.clip_param,
            vf_loss_coeff=train_cfg.vf_loss_coeff,
            entropy_coeff=train_cfg.entropy_coeff,
            train_batch_size=train_cfg.train_batch_size,
            sgd_minibatch_size=train_cfg.sgd_minibatch_size,
            num_sgd_iter=train_cfg.num_sgd_iter,
            model=train_cfg.model,
            # Memory optimization: use lower float precision if enabled
            _use_trajectory_view_api=True,  # More memory efficient
        )
        .evaluation(
            evaluation_interval=train_cfg.evaluation_interval,
            evaluation_duration=train_cfg.evaluation_duration,
            evaluation_num_workers=train_cfg.evaluation_num_workers,
            evaluation_parallel_to_training=train_cfg.evaluation_parallel_to_training,
            evaluation_config=PPOConfig.overrides(**train_cfg.evaluation_config)
        )
        .debugging(log_level=train_cfg.log_level)
        # Add fault tolerance: save checkpoints frequently
        .checkpoint(
            checkpoint_frequency=train_cfg.checkpoint_freq,
            checkpoint_at_end=train_cfg.checkpoint_at_end,
            keep_checkpoints_num=train_cfg.keep_checkpoints_num
        )
    )
    
    # Create output directory
    exp_dir = Path(train_cfg.local_dir) / train_cfg.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration for reproducibility
    with open(exp_dir / "train_config.json", "w") as f:
        config_dict = {k: v for k, v in train_cfg.__dict__.items() 
                      if not k.startswith("_") and not callable(v)}
        json.dump(config_dict, f, indent=2, default=str)
    
    # Configure logger callback
    callbacks = [
        tune.logger.TBXLoggerCallback(),
    ]
    
    # Add WandB callback if requested
    if hasattr(train_cfg, 'wandb_project') and train_cfg.wandb_project:
        try:
            from loggers import WandbLoggerCallback
            callbacks.append(
                WandbLoggerCallback(
                    project=train_cfg.wandb_project,
                    log_config=True,
                )
            )
            logger.info(f"WandB logging enabled for project: {train_cfg.wandb_project}")
        except ImportError:
            logger.warning("WandB logger requested but not available")
    
    # Configure stopping criteria
    stop = {
        "training_iteration": train_cfg.num_iterations,
    }
    
    # Start training with memory monitoring
    logger.info(f"Starting training for {train_cfg.num_iterations} iterations")
    logger.info(f"Results will be saved to: {exp_dir}")
    
    # Create memory tracker if available
    if MemoryTracker:
        memory_tracker = MemoryTracker(log_dir=exp_dir / "memory_logs")
        memory_tracker.start()
    else:
        memory_tracker = None
    
    try:
        # Run training
        analysis = tune.run(
            "PPO",
            name=train_cfg.tune_exp_name,
            config=algo_config.to_dict(),
            stop=stop,
            local_dir=train_cfg.local_dir,
            checkpoint_freq=train_cfg.checkpoint_freq,
            checkpoint_at_end=train_cfg.checkpoint_at_end,
            keep_checkpoints_num=train_cfg.keep_checkpoints_num,
            callbacks=callbacks,
            verbose=1,
            # Add fault tolerance
            max_failures=3,  # Allow restarts on failure
            restore=True,    # Try to restore from previous runs
        )
        
        logger.info("Training completed successfully")
        
        # Get best checkpoint
        best_checkpoint = get_best_checkpoint(analysis)
        
        # Save the path to best checkpoint
        if best_checkpoint:
            with open(exp_dir / "best_checkpoint.txt", "w") as f:
                f.write(f"{best_checkpoint}\n")
        
        return analysis
    
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return None
    
    finally:
        # Stop memory tracking
        if memory_tracker:
            memory_tracker.stop()
            logger.info(memory_tracker.get_memory_summary())
        
        # Clean up Ray
        if ray.is_initialized():
            logger.info("Shutting down Ray")
            ray.shutdown()
        
        # Force garbage collection
        gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Train a PPO agent for Poker.")
    parser.add_argument("--workers", type=int, help="Override number of workers")
    parser.add_argument("--batch_size", type=int, help="Override train batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--iterations", type=int, help="Override number of iterations")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--memory-efficient", action="store_true", 
                       help="Enable memory-efficient mode (reduces batch size and workers)")
    args = parser.parse_args()
    
    # Set log level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration instances
    poker_config = PokerConfig()
    training_config = TrainingConfig()
    
    # Override config from arguments
    if args.workers is not None:
        training_config.num_workers = args.workers
        
    if args.batch_size is not None:
        training_config.train_batch_size = args.batch_size
        # Adjust SGD batch size too
        training_config.sgd_minibatch_size = min(
            training_config.sgd_minibatch_size, 
            training_config.train_batch_size // 4
        )
        
    if args.lr is not None:
        training_config.lr = args.lr
        
    if args.iterations is not None:
        training_config.num_iterations = args.iterations
    
    # Apply memory-efficient mode if requested
    if args.memory_efficient:
        logger.info("Memory-efficient mode enabled")
        training_config.num_workers = max(1, min(4, training_config.num_workers))
        training_config.train_batch_size = min(4096, training_config.train_batch_size)
        training_config.sgd_minibatch_size = min(1024, training_config.sgd_minibatch_size)
        training_config.rollout_fragment_length = min(100, training_config.rollout_fragment_length)
    
    # Log configuration
    logger.info("=== Poker Configuration ===")
    for key, value in vars(poker_config).items():
        if not key.startswith("_"):
            logger.info(f"{key}: {value}")
    
    logger.info("=== Training Configuration ===")
    for key, value in vars(training_config).items():
        if not key.startswith("_"):
            logger.info(f"{key}: {value}")
    
    # Start training
    try:
        start_time = time.time()
        analysis = train_poker(poker_config, training_config)
        duration = time.time() - start_time
        
        if analysis:
            logger.info(f"Training completed in {duration:.2f} seconds")
        else:
            logger.error("Training failed or was interrupted")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        # Clean up Ray if it's still running
        if ray.is_initialized():
            logger.info("Shutting down Ray due to error")
            ray.shutdown()

if __name__ == "__main__":
    main()

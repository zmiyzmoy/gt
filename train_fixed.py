import argparse
import functools
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from functools import wraps
from typing import Dict, Any, Optional, Union

import numpy as np
import psutil
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import Tuner

try:
    # Попытка импорта из локальных файлов, с обработкой ошибок
    from config import PokerConfig, TrainingConfig
    from environment_fixed import PokerEnv, create_poker_env
    from loggers import WandbLoggerCallback
    # Регистрация моделей
    from register_models import register_models
    
    # Регистрируем модели перед использованием
    register_models()
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all required files are in the correct locations.")
    sys.exit(1)

# Настройка логирования
logger = logging.getLogger("main")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def monitor_training_memory(func):
    """
    Декоратор для мониторинга использования памяти во время обучения.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / 1024 / 1024  # МБ
        start_time = time.time()
        
        logger.info(f"Starting {func.__name__} with memory usage: {start_mem:.2f} MB")
        
        try:
            result = func(*args, **kwargs)
            
            end_mem = process.memory_info().rss / 1024 / 1024  # МБ
            elapsed_time = time.time() - start_time
            
            logger.info(f"Finished {func.__name__} in {elapsed_time:.2f} seconds")
            logger.info(f"Memory usage: {end_mem:.2f} MB (delta: {end_mem - start_mem:.2f} MB)")
            
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    
    return wrapper

def create_env(env_config):
    """Фабричная функция для создания экземпляров среды."""
    try:
        return PokerEnv(env_config)
    except Exception as e:
        logger.error(f"Error creating environment: {e}")
        logger.error(traceback.format_exc())
        raise

def get_available_resources():
    """Получает информацию о доступных системных ресурсах."""
    cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count()
    total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
    
    # Проверка наличия GPU через ray
    try:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        gpu_ids = ray.get_gpu_ids()
        num_gpus = len(gpu_ids)
    except Exception:
        num_gpus = 0
    
    return {
        "cpu_count": cpu_count,
        "total_memory_gb": total_memory_gb,
        "num_gpus": num_gpus
    }

def optimize_config_for_resources(train_cfg: TrainingConfig):
    """Оптимизирует конфигурацию обучения на основе доступных ресурсов."""
    resources = get_available_resources()
    
    # Расчет оптимального количества воркеров
    cpu_count = resources["cpu_count"]
    memory_gb = resources["total_memory_gb"]
    num_gpus = resources["num_gpus"]
    
    logger.info(f"Available resources: {cpu_count} CPU cores, {memory_gb:.2f} GB RAM, {num_gpus} GPUs")
    
    # Резервируем 1 CPU для основного потока и системы
    available_cpus = max(1, cpu_count - 1)
    
    # Настройка для CPU-only
    if num_gpus == 0:
        logger.info("No GPUs detected, optimizing for CPU-only training")
        # Уменьшаем размер batch и количество SGD итераций для CPU
        train_cfg.train_batch_size = min(train_cfg.train_batch_size, 2048)
        train_cfg.sgd_minibatch_size = min(train_cfg.sgd_minibatch_size, 512)
        train_cfg.num_sgd_iter = min(train_cfg.num_sgd_iter, 5)
        
        # Ограничиваем количество воркеров на маломощных машинах
        if memory_gb < 8:
            logger.warning("Low memory detected, reducing resource usage")
            train_cfg.num_workers = min(2, available_cpus)
            train_cfg.train_batch_size = min(train_cfg.train_batch_size, 1024)
        else:
            train_cfg.num_workers = min(train_cfg.num_workers, available_cpus)
    else:
        # С GPU можем использовать больше воркеров и большие батчи
        train_cfg.num_gpus = min(train_cfg.num_gpus, num_gpus)
        if train_cfg.num_gpus > 0:
            # Резервируем GPU для основного процесса
            available_gpus = num_gpus - train_cfg.num_gpus
            if available_gpus > 0:
                train_cfg.num_gpus_per_worker = available_gpus / train_cfg.num_workers
    
    # Ограничиваем количество env на worker для экономии памяти
    train_cfg.num_envs_per_worker = min(train_cfg.num_envs_per_worker, 2)
    
    logger.info(f"Optimized configuration: workers={train_cfg.num_workers}, "
                f"batch_size={train_cfg.train_batch_size}, "
                f"sgd_batch={train_cfg.sgd_minibatch_size}, "
                f"sgd_iter={train_cfg.num_sgd_iter}")
    
    return train_cfg

def setup_ray(log_level="INFO"):
    """Инициализирует Ray с обработкой ошибок."""
    try:
        # Проверяем, инициализирован ли Ray
        if ray.is_initialized():
            ray.shutdown()
        
        # Инициализируем Ray
        ray.init(
            logging_level=log_level,
            log_to_driver=True,
            ignore_reinit_error=True,
        )
        
        logger.info("Ray initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Ray: {e}")
        logger.error(traceback.format_exc())
        return False

def get_best_checkpoint(analysis, metric="episode_reward_mean"):
    """Безопасно получает лучший чекпоинт из результатов анализа."""
    try:
        if analysis and hasattr(analysis, "get_best_checkpoint"):
            return analysis.get_best_checkpoint(metric=metric, mode="max")
        logger.warning("Analysis object doesn't support get_best_checkpoint method")
        return None
    except Exception as e:
        logger.error(f"Error getting best checkpoint: {e}")
        return None

@monitor_training_memory
def train_poker(poker_cfg: PokerConfig, train_cfg: TrainingConfig):
    """
    Основная функция обучения.
    
    Args:
        poker_cfg: Конфигурация покерной среды
        train_cfg: Конфигурация процесса обучения
    """
    # Инициализация Ray
    if not setup_ray(train_cfg.log_level):
        logger.error("Failed to initialize Ray. Exiting.")
        return
    
    try:
        # Оптимизация конфигурации под доступные ресурсы
        train_cfg = optimize_config_for_resources(train_cfg)
        
        # Формирование конфигурации среды из poker_cfg
        env_config = {
            "game_name": poker_cfg.game_name,
            "num_players": poker_cfg.num_players,
            "starting_stack": poker_cfg.starting_stack,
            "small_blind": poker_cfg.small_blind,
            "big_blind": poker_cfg.big_blind,
            "game_config": poker_cfg.game_config,
        }
        
        # Создание и валидация тестовой среды
        logger.info("Creating test environment to validate configuration")
        test_env = create_env(env_config)
        observation_space = test_env.observation_space
        action_space = test_env.action_space
        logger.info(f"Environment initialized with: obs_space={observation_space}, action_space={action_space}")
        
        # Получение размера наблюдения и количества действий
        if hasattr(observation_space, "spaces") and "obs" in observation_space.spaces:
            obs_size = int(np.prod(observation_space.spaces["obs"].shape))
        else:
            obs_size = int(np.prod(observation_space.shape))
        
        num_actions = action_space.n
        logger.info(f"Observation size: {obs_size}, Number of actions: {num_actions}")
        
        # Подготовка callbacks для обучения
        callbacks = []
        if train_cfg.wandb_project:
            callbacks.append(
                WandbLoggerCallback(
                    project=train_cfg.wandb_project,
                    log_config=True,
                )
            )
        
        # Обновление конфигурации модели с правильными размерностями
        if poker_cfg.model_config.get("custom_model"):
            model_config = poker_cfg.model_config.copy()
            # Добавляем размерности в конфигурацию модели, если они отсутствуют
            if "custom_model_config" not in model_config:
                model_config["custom_model_config"] = {}
            model_config["custom_model_config"]["obs_size"] = obs_size
            model_config["custom_model_config"]["num_actions"] = num_actions
        else:
            model_config = {}
        
        # Создание конфигурации алгоритма PPO
        config = (
            PPOConfig()
            # Environment
            .environment(
                env=PokerEnv,
                env_config=env_config,
            )
            # Model
            .training(
                model={
                    **model_config,
                    **train_cfg.model,  # Добавление дополнительных параметров модели
                },
                lr=train_cfg.lr,
                gamma=train_cfg.gamma,
                lambda_=train_cfg.lambda_,
                clip_param=train_cfg.clip_param,
                vf_loss_coeff=train_cfg.vf_loss_coeff,
                entropy_coeff=train_cfg.entropy_coeff,
                train_batch_size=train_cfg.train_batch_size,
                sgd_minibatch_size=train_cfg.sgd_minibatch_size,
                num_sgd_iter=train_cfg.num_sgd_iter,
                rollout_fragment_length=train_cfg.rollout_fragment_length,
                batch_mode=train_cfg.batch_mode,
            )
            # Resources
            .resources(
                num_gpus=train_cfg.num_gpus,
                num_cpus_per_worker=train_cfg.num_cpus_per_worker,
                num_gpus_per_worker=train_cfg.num_gpus_per_worker,
                num_workers=train_cfg.num_workers,
                num_envs_per_worker=train_cfg.num_envs_per_worker,
            )
            # Evaluation (если настроено)
            .evaluation(
                evaluation_interval=train_cfg.evaluation_interval if train_cfg.evaluation_interval > 0 else None,
                evaluation_duration=train_cfg.evaluation_duration,
                evaluation_num_workers=train_cfg.evaluation_num_workers,
                evaluation_parallel_to_training=train_cfg.evaluation_parallel_to_training,
                evaluation_config=train_cfg.evaluation_config,
            )
            # Reporting & Checkpointing
            .reporting(
                min_time_s_per_iteration=10,  # Минимальное время на итерацию для стабильных отчетов
                metrics_num_episodes_for_smoothing=100,  # Сглаживание метрик
            )
            .checkpointing(
                checkpoint_frequency=train_cfg.checkpoint_freq,
                checkpoint_at_end=train_cfg.checkpoint_at_end,
                keep_checkpoints_num=train_cfg.keep_checkpoints_num,
            )
            # Debugging & Logging
            .debugging(
                log_level=train_cfg.log_level,
                logger_config={
                    "logdir": train_cfg.local_dir,
                    "type": "ray.tune.logger.TBXLogger",
                }
            )
            .framework("torch")
            .build()
        )
        
        # Логирование итоговой конфигурации
        logger.info(f"Starting training with configuration: {json.dumps(config.to_dict(), indent=2)}")
        
        # Запуск обучения
        analysis = tune.run(
            "PPO",
            name=train_cfg.exp_name,
            config=config.to_dict(),
            stop={"training_iteration": train_cfg.num_iterations},
            checkpoint_freq=train_cfg.checkpoint_freq,
            checkpoint_at_end=train_cfg.checkpoint_at_end,
            keep_checkpoints_num=train_cfg.keep_checkpoints_num,
            verbose=1,
            local_dir=train_cfg.local_dir,
            callbacks=callbacks,
        )
        
        # Получение лучшего чекпоинта
        best_checkpoint = get_best_checkpoint(analysis)
        if best_checkpoint:
            logger.info(f"Best checkpoint path: {best_checkpoint}")
            
            # Сохранение пути к лучшему чекпоинту в отдельный файл для удобства
            checkpoint_info_path = os.path.join(train_cfg.local_dir, "best_checkpoint_info.json")
            with open(checkpoint_info_path, "w") as f:
                json.dump({"best_checkpoint": best_checkpoint}, f)
            
            logger.info(f"Best checkpoint info saved to: {checkpoint_info_path}")
        else:
            logger.warning("No best checkpoint found.")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.error(traceback.format_exc())
        # Гарантируем завершение Ray
        ray.shutdown()
        logger.info("Ray shutdown completed.")
        raise
    finally:
        # Гарантируем завершение Ray в любом случае
        try:
            ray.shutdown()
            logger.info("Ray shutdown completed.")
        except Exception as e:
            logger.error(f"Error shutting down Ray: {e}")

def main():
    """Основная точка входа скрипта."""
    parser = argparse.ArgumentParser(description="Train a poker agent using RLlib.")
    parser.add_argument("--config", type=str, default="config.py", help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--iterations", type=int, help="Override number of training iterations")
    args = parser.parse_args()
    
    # Настройка уровня логирования
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    try:
        # Загрузка конфигурации (уже импортирована выше)
        poker_config = PokerConfig()
        training_config = TrainingConfig()
        
        # Переопределение параметров из аргументов
        if args.iterations:
            training_config.num_iterations = args.iterations
        
        # Запуск обучения
        logger.info("Starting training process")
        train_poker(poker_config, training_config)
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training process failed.\n{e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
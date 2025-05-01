# train.py
import argparse
import os
import logging
from pathlib import Path
import numpy as np

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print, TBXLoggerCallback
# Импортируем конфиги и среду
from config import PokerConfig, TrainingConfig
from environment import PokerEnv
from loggers import WandbLoggerCallback # Импортируем, но пока не используем
from models import AdvancedPokerModel

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_env(env_config):
    """Фабричная функция для создания экземпляра среды."""
    return PokerEnv(env_config)

def train_poker(poker_cfg: PokerConfig, train_cfg: TrainingConfig):
    """Основная функция для настройки и запуска обучения."""
    logger.info("Starting training setup...")

    # Регистрация кастомной модели
    try:
        ModelCatalog.register_custom_model("AdvancedPokerModel", AdvancedPokerModel)
        logger.info(f"Custom model '{poker_cfg.model_config['custom_model']}' registered successfully.")
    except Exception as e:
         logger.error(f"Failed to register custom model: {e}", exc_info=True)
         raise

    # Настройка Ray
    if not ray.is_initialized():
        try:
            ray.init(
                logging_level=train_cfg.log_level,
                log_to_driver=True # Выводить логи воркеров в драйвер
            )
            logger.info("Ray initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}", exc_info=True)
            raise

    # Конфигурация алгоритма PPO
    env_creator_config = {"config": poker_cfg, "dtype": np.float32} # Конфиг для создания среды

    # --- Используем корректную конфигурацию для Ray 2.10.0 ---
    algo_config = (
        PPOConfig()
        .environment(env=PokerEnv, env_config=env_creator_config)
        .framework("torch")
        .resources( # Ресурсы для тренера И для каждого воркера
            num_gpus=train_cfg.num_gpus, # GPU для тренера (основного процесса)
            num_cpus_per_worker=train_cfg.num_cpus_per_worker, # CPU НА воркер
            num_gpus_per_worker=train_cfg.num_gpus_per_worker, # GPU НА воркер
        )
        .rollouts( # Количество воркеров и параметры сбора данных
            num_rollout_workers=train_cfg.num_workers, # КОЛИЧЕСТВО воркеров
            num_envs_per_worker=train_cfg.num_envs_per_worker,
            rollout_fragment_length=train_cfg.rollout_fragment_length,
            batch_mode=train_cfg.batch_mode,
            # enable_connectors=True # Включить для нового API коннекторов (если используется)
        )
        .training( # Параметры обучения
            gamma=train_cfg.gamma,
            lr=train_cfg.lr,
            lambda_=train_cfg.lambda_,
            clip_param=train_cfg.clip_param,
            vf_loss_coeff=train_cfg.vf_loss_coeff,
            entropy_coeff=train_cfg.entropy_coeff,
            train_batch_size=train_cfg.train_batch_size,
            sgd_minibatch_size=train_cfg.sgd_minibatch_size,
            num_sgd_iter=train_cfg.num_sgd_iter,
            model=train_cfg.model, # Передаем конфиг модели из TrainingConfig
        )
        .evaluation( # Параметры оценки
            evaluation_interval=train_cfg.evaluation_interval,
            evaluation_duration=train_cfg.evaluation_duration,
            evaluation_num_workers=train_cfg.evaluation_num_workers, # Количество воркеров для оценки
            evaluation_parallel_to_training=train_cfg.evaluation_parallel_to_training,
            # Ресурсы для evaluation workers наследуются из .resources() по умолчанию.
            # Передаем evaluation_config через `overrides`
            evaluation_config=PPOConfig.overrides(
                 # Если нужны ДРУГИЕ ресурсы для evaluation, раскомментируйте и задайте:
                 # num_cpus_per_worker=...,
                 # num_gpus_per_worker=...,
                 **train_cfg.evaluation_config # Добавляем остальные параметры (explore=False)
             )
        )
        .debugging(log_level=train_cfg.log_level) # Уровень логов RLlib
    )
    # --- КОНЕЦ КОНФИГУРАЦИИ ---

    # Создание директории для эксперимента
    exp_dir = Path(train_cfg.local_dir) / train_cfg.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Experiment results will be saved to: {exp_dir}")

    # Настройка логгеров для Ray Tune (Wandb пока отключен)
    callbacks = [
        TBXLoggerCallback(), # Логгер для TensorBoard
        # WandbLoggerCallback(                 # <<< Wandb пока отключен
        #     project=train_cfg.wandb_project,
        #     # api_key_file=...,
        #     log_config=True,
        #     config_dict={**poker_cfg.__dict__, **train_cfg.__dict__}
        # )
    ]

    # Параметры остановки обучения
    stop = {
        "training_iteration": train_cfg.num_iterations,
        # Можно добавить другие критерии остановки, например, по награде:
        # "episode_reward_mean": 1000, # Остановить, когда средняя награда достигнет 1000
    }

    # Запуск эксперимента Ray Tune
    logger.info(f"Starting Ray Tune experiment: {train_cfg.tune_exp_name}")
    try:
        analysis = tune.run(
            "PPO", # Имя алгоритма
            name=train_cfg.tune_exp_name, # Имя эксперимента
            config=algo_config.to_dict(), # Конфиг алгоритма
            stop=stop, # Условия остановки
            local_dir=train_cfg.local_dir, # Базовая директория для результатов
            checkpoint_freq=train_cfg.checkpoint_freq,
            checkpoint_at_end=train_cfg.checkpoint_at_end,
            keep_checkpoints_num=train_cfg.keep_checkpoints_num,
            callbacks=callbacks, # Используемые логгеры/коллбэки
            verbose=1 # Уровень вывода Tune (1 - таблица статуса)
        )
        logger.info("Training finished or interrupted.")

        # --- Используем корректный API для получения результатов в Ray 2.10 ---
        if analysis.trials:
            # Определяем метрику для выбора лучшего trial
            default_metric = "episode_reward_mean"
            eval_metric = "evaluation/episode_reward_mean"
            metric_to_use = default_metric # По умолчанию используем награду обучения

            # Пытаемся найти лучший trial по метрике оценки, если она есть
            best_eval_trial = analysis.get_best_trial(eval_metric, mode="max", scope="last")
            if best_eval_trial:
                 metric_to_use = eval_metric
                 best_trial = best_eval_trial
                 logger.info(f"Using evaluation metric '{metric_to_use}' for best trial selection.")
            else:
                 # Если нет результатов оценки, используем метрику обучения
                 best_trial = analysis.get_best_trial(default_metric, mode="max", scope="last")
                 logger.info(f"Evaluation metric '{eval_metric}' not found or no evaluation run. Using training metric '{metric_to_use}'.")

            if best_trial:
                 logger.info(f"Best trial found: {best_trial.trial_id}")
                 logger.info(f"Best trial config: {pretty_print(best_trial.config)}")
                 last_result = best_trial.last_result
                 if last_result:
                      logger.info(f"Best trial final metrics ({metric_to_use}): {last_result.get(metric_to_use, 'N/A')}")
                      # logger.info(f"Full last metrics: {pretty_print(last_result)}") # Раскомментировать для полного вывода
                 else:
                      logger.warning("Could not get last result for the best trial.")

                 # Получаем лучший чекпоинт для этого trial
                 best_checkpoint_result = analysis.get_best_checkpoint(
                     trial=best_trial,
                     metric=metric_to_use,
                     mode="max"
                 )
                 if best_checkpoint_result:
                      checkpoint_path = getattr(best_checkpoint_result, 'path', str(best_checkpoint_result))
                      logger.info(f"Best trial checkpoint path: {checkpoint_path}")
                 else:
                      logger.warning(f"Could not retrieve best checkpoint for trial {best_trial.trial_id} based on {metric_to_use}.")
            else:
                 logger.warning(f"Could not determine the best trial based on metric '{metric_to_use}'.")
        else:
            logger.warning("No trials were completed.")
        # --- КОНЕЦ НОВОГО КОДА для получения результатов ---

        return analysis

    except Exception as e:
        logger.error(f"\nCritical error during tune.run: {e}", exc_info=True)
        if isinstance(e, tune.error.TuneError):
             logger.error(f"TuneError details: {e.args}")
        raise # Перевыбрасываем исключение, чтобы основной блок его поймал

    finally:
        logger.info("\n=== Cleanup ===")
        if ray.is_initialized():
             ray.shutdown()
             logger.info("Ray shutdown completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO agent for Poker.")
    # Добавьте сюда аргументы командной строки, если нужно переопределять конфиги
    # parser.add_argument("--lr", type=float, help="Override learning rate")
    args = parser.parse_args()

    # Создание экземпляров конфигов
    poker_config = PokerConfig()
    training_config = TrainingConfig()

    # Переопределение конфигов из аргументов (пример)
    # if args.lr is not None: training_config.lr = args.lr

    logger.info("--- Poker Configuration ---")
    logger.info(pretty_print(poker_config.__dict__))
    logger.info("--- Training Configuration ---")
    logger.info(pretty_print(training_config.__dict__))

    try:
        train_poker(poker_config, training_config)
        logger.info("Training process completed successfully.")
    except Exception as e:
        logger.error("Training process failed.", exc_info=True) # Логируем с traceback
        # Завершаем Ray, если он еще работает
        if ray.is_initialized():
            logger.info("Shutting down Ray due to error...")
            ray.shutdown()
        exit(1) # Выход с кодом ошибки

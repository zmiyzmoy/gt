# train.py
import argparse
import os
import logging
from pathlib import Path
import numpy as np

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog # <<< ИМПОРТ ДЛЯ РЕГИСТРАЦИИ
from ray.tune.logger import pretty_print, Logger, TBXLoggerCallback # Добавим TensorBoard логгер
# Импортируем конфиги и среду
from config import PokerConfig, TrainingConfig
from environment import PokerEnv
from loggers import WandbLoggerCallback # Ваш логгер W&B
from models import AdvancedPokerModel # <<< ИМПОРТ КАСТОМНОЙ МОДЕЛИ

# Настройка логирования для основного скрипта
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_env(env_config):
    """Фабричная функция для создания экземпляра среды."""
    # env_config будет содержать {'config': PokerConfig(...), 'dtype': np.float32}
    # или то, что передано через algo_config.environment()
    return PokerEnv(env_config)

def train_poker(poker_cfg: PokerConfig, train_cfg: TrainingConfig):
    """Основная функция для настройки и запуска обучения."""
    logger.info("Starting training setup...")

    # --- ИСПРАВЛЕНО: Регистрация кастомной модели ---
    try:
        ModelCatalog.register_custom_model("AdvancedPokerModel", AdvancedPokerModel)
        logger.info(f"Custom model '{poker_cfg.model_config['custom_model']}' registered successfully.")
    except Exception as e:
         logger.error(f"Failed to register custom model: {e}", exc_info=True)
         raise # Прерываем выполнение, если модель не зарегистрирована

    # Настройка Ray
    if not ray.is_initialized():
        try:
            # Попробуем явно указать каталог для логов ray
            # runtime_env = {"working_dir": str(Path(__file__).parent.resolve())} # Если нужно указать рабочую директорию
            ray.init(
                # dashboard_host='0.0.0.0', # Если нужен доступ к dashboard извне
                # runtime_env=runtime_env,
                logging_level=train_cfg.log_level, # Уровень логов Ray
                log_to_driver=True # Выводить логи воркеров в драйвер
            )
            logger.info("Ray initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}", exc_info=True)
            raise

    # Конфигурация алгоритма PPO
    env_creator_config = {"config": poker_cfg, "dtype": np.float32} # Передаем конфиг игры и dtype

    algo_config = (
        PPOConfig()
        .environment(
             env=PokerEnv, # Используем класс среды
             env_config=env_creator_config # Передаем конфиг для env_creator
        )
        .framework("torch") # Используем PyTorch
        .resources(
            num_gpus=train_cfg.num_gpus,
            num_workers=train_cfg.num_workers,
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
            model=train_cfg.model, # Передаем конфиг модели ИЗ TrainingConfig
        )
        .evaluation(
            evaluation_interval=train_cfg.evaluation_interval,
            evaluation_duration=train_cfg.evaluation_duration,
            evaluation_num_workers=train_cfg.evaluation_num_workers,
            evaluation_parallel_to_training=train_cfg.evaluation_parallel_to_training,
            evaluation_config=train_cfg.evaluation_config # Передаем evaluation_config
            # Можно также передать env_config для оценки, если он отличается
            # .evaluation(..., evaluation_config=AlgorithmConfig.overrides(env_config=eval_env_config))
        )
        .debugging(log_level=train_cfg.log_level) # Уровень логов RLlib
    )

    # Создание директории для эксперимента
    exp_dir = Path(train_cfg.local_dir) / train_cfg.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Experiment results will be saved to: {exp_dir}")

    # Настройка логгеров для Ray Tune
    callbacks = [
        # Логгер по умолчанию (выводит в консоль и файлы json/csv)
        # BetterLogger(log_dir=str(exp_dir)), # Заменен стандартным выводом Tune
        # Логгер для TensorBoard
        TBXLoggerCallback(),
        # Ваш кастомный логгер W&B
        WandbLoggerCallback(
            project=train_cfg.wandb_project,
            # api_key_file=train_cfg.wandb_api_key_file, # Если используете файл
            # api_key="YOUR_API_KEY", # Или передаете ключ напрямую
            log_config=True,
            config={**poker_cfg.__dict__, **train_cfg.__dict__} # Логируем оба конфига
        )
    ]

    # Параметры остановки обучения
    stop = {
        "training_iteration": train_cfg.num_iterations,
    }

    # Запуск эксперимента Ray Tune
    logger.info(f"Starting Ray Tune experiment: {train_cfg.tune_exp_name}")
    try:
        analysis = tune.run(
            "PPO", # Используем зарегистрированный PPO
            name=train_cfg.tune_exp_name, # Имя эксперимента в Tune
            config=algo_config.to_dict(), # Передаем конфиг как словарь
            stop=stop,
            local_dir=train_cfg.local_dir, # Базовая директория для всех экспериментов
            checkpoint_freq=train_cfg.checkpoint_freq,
            checkpoint_at_end=train_cfg.checkpoint_at_end,
            keep_checkpoints_num=train_cfg.keep_checkpoints_num,
            # resources_per_trial=PPO.default_resource_request(algo_config), # Ray обычно определяет ресурсы сам
            callbacks=callbacks,
            verbose=1 # 0 = тихо, 1 = статус + таблица, 2 = подробно, 3 = очень подробно
        )
        logger.info("Training finished.")
        # Вывод лучшего результата
        best_trial = analysis.get_best_trial("episode_reward_mean", mode="max", scope="last")
        if best_trial:
             logger.info(f"Best trial config: {pretty_print(best_trial.config)}")
             logger.info(f"Best trial final validation reward: {best_trial.last_result['episode_reward_mean']}")
             logger.info(f"Best trial checkpoint path: {analysis.get_best_checkpoint(best_trial, metric='episode_reward_mean', mode='max')}")
        else:
             logger.warning("Could not determine the best trial.")

        return analysis

    except Exception as e:
        logger.error(f"\nCritical error during training: {e}", exc_info=True)
        # Попытка поймать TuneError для более специфичного сообщения
        if isinstance(e, tune.error.TuneError):
             logger.error(f"TuneError details: {e.args}")
        # Поднять исключение дальше, чтобы скрипт завершился с ошибкой
        raise

    finally:
        logger.info("\n=== Cleanup ===")
        ray.shutdown()
        logger.info("Ray shutdown completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO agent for Poker.")
    # Можно добавить аргументы для переопределения конфигов через командную строку
    # parser.add_argument("--lr", type=float, help="Override learning rate")
    # ...

    args = parser.parse_args()

    # Создание экземпляров конфигов
    poker_config = PokerConfig()
    training_config = TrainingConfig()

    # Переопределение конфигов из аргументов командной строки (если нужно)
    # if args.lr:
    #     training_config.lr = args.lr
    # ...

    logger.info("--- Poker Configuration ---")
    logger.info(pretty_print(poker_config.__dict__))
    logger.info("--- Training Configuration ---")
    logger.info(pretty_print(training_config.__dict__))

    try:
        train_poker(poker_config, training_config)
        logger.info("Training process completed successfully.")
    except Exception as e:
        logger.error("Training process failed.", exc_info=False) # Не выводим Traceback второй раз
        exit(1) # Выход с кодом ошибки

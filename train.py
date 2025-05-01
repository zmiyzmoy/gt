# train.py
import os
import sys
import logging
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
# import wandb # Раскомментируй, если используешь W&B
# from ray.tune.logger import WandbLoggerCallback # Импорт для W&B в Ray 2.10
import numpy as np
from pathlib import Path

# Импортируем наши модули
from config import PokerConfig, TrainingConfig
from environment import PokerEnv
from models import AdvancedPokerModel

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('poker_training.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Устанавливаем режим W&B (если используется)
# if not os.environ.get("WANDB_API_KEY"):
#     os.environ["WANDB_MODE"] = "offline"

def setup_wandb(config: TrainingConfig):
    """Настройка Weights & Biases (опционально)"""
    # if config.wandb_project and os.environ.get("WANDB_MODE") != "disabled":
    #     try:
    #         wandb.init(...) # Настройка W&B init
    #         logger.info("W&B initialized...")
    #         return True
    #     except Exception as e:
    #         logger.warning(f"Failed to initialize W&B: {e}. Disabling.")
    #         os.environ["WANDB_MODE"] = "disabled"
    return False # Возвращаем False, если W&B не используется


def create_env(env_config):
    """Фабрика для создания экземпляра среды."""
    return PokerEnv(env_config)

def train_poker():
    """Основная функция обучения."""
    run_successful = False
    try:
        logger.info("\n=== Starting poker training ===")

        # Инициализируем конфигурации
        poker_config = PokerConfig()
        training_config = TrainingConfig()
        training_config.model = poker_config.model_config # Связываем модель

        # Настраиваем W&B (если нужно)
        use_wandb = setup_wandb(training_config)

        # --- Инициализируем Ray ---
        available_cpus = os.cpu_count() or 12
        # Корректируем num_workers на основе доступных CPU
        num_workers = min(training_config.num_workers, available_cpus - 2) # -2: для драйвера и резерва
        if num_workers != training_config.num_workers:
             logger.warning(f"Reduced num_workers from {training_config.num_workers} to {num_workers} based on available CPUs ({available_cpus})")

        ray.init(
            num_cpus=num_workers + 1, # Указываем точное кол-во CPU для Ray
            num_gpus=training_config.num_gpus,
            logging_level=logging.INFO,
            ignore_reinit_error=True
        )
        logger.info("Ray initialized. View dashboard at http://127.0.0.1:8265 (default address)")


        # Регистрируем окружение
        register_env("PokerEnv", create_env)

        # --- Настраиваем PPOConfig для Ray 2.10.0 ---
        logger.info("Configuring PPO algorithm for Ray 2.10.0...")
        ppo_config_builder = (
            PPOConfig()
            .environment(env="PokerEnv", env_config={"config": poker_config})
            .framework("torch")
            .training(
                gamma=training_config.gamma,
                lr=training_config.lr,
                lambda_=training_config.lambda_,
                clip_param=training_config.clip_param,
                vf_loss_coeff=training_config.vf_loss_coeff,
                entropy_coeff=training_config.entropy_coeff,
                train_batch_size=training_config.train_batch_size,
                sgd_minibatch_size=training_config.sgd_minibatch_size,
                num_sgd_iter=training_config.num_sgd_iter,
                model=training_config.model
            )
            .rollouts(
                num_rollout_workers=num_workers, # Используем скорректированное значение
                num_envs_per_worker=training_config.num_envs_per_worker,
                rollout_fragment_length=training_config.rollout_fragment_length,
                batch_mode=training_config.batch_mode
            )
            .resources(
                num_gpus=training_config.num_gpus,
                num_cpus_per_worker=training_config.num_cpus_per_worker,
                num_gpus_per_worker=training_config.num_gpus_per_worker
            )
            .evaluation(
                evaluation_interval=training_config.evaluation_interval,
                evaluation_duration=training_config.evaluation_duration,
                evaluation_num_workers=training_config.evaluation_num_workers,
                evaluation_parallel_to_training=training_config.evaluation_parallel_to_training,
                evaluation_config={"explore": False}
            )
            .debugging(log_level=training_config.log_level)
            # .callbacks(PokerCallbacks) # Раскомментируй, если нужны твои колбэки
        )
        final_ppo_config = ppo_config_builder.to_dict()
        logger.info("PPOConfig configured.")


        # --- Запускаем обучение с Ray Tune (classic API) ---
        logger.info(f"Starting Ray Tune experiment '{training_config.tune_exp_name}'...")
        local_dir_path = Path(training_config.local_dir)
        # Tune ожидает родительскую папку в local_dir для классического API
        storage_parent_path = str(local_dir_path.parent.resolve())
        exp_dir_name = local_dir_path.name # Имя папки будет создано внутри storage_parent_path

        logger.info(f"Results will be stored under: {storage_parent_path}")
        # Tune сам создаст папку эксперимента, не нужно делать mkdir

        tune_callbacks = []
        # if use_wandb:
        #      try:
        #           from ray.tune.logger import WandbLogger
        #           tune_callbacks.append(WandbLogger(project=training_config.wandb_project))
        #           logger.info("Using WandbLogger for Ray Tune.")
        #      except ImportError:
        #           logger.warning("Could not import WandbLogger. W&B Tune callback disabled.")

        analysis = tune.run(
            "PPO",
            name=exp_dir_name, # Имя папки эксперимента
            config=final_ppo_config,
            stop={"training_iteration": training_config.num_iterations},
            local_dir=storage_parent_path, # Родительская папка для результатов
            checkpoint_freq=training_config.checkpoint_freq,
            checkpoint_at_end=training_config.checkpoint_at_end,
            keep_checkpoints_num=training_config.keep_checkpoints_num,
            verbose=1,
            # fail_fast=True, # Можно раскомментировать
            # max_failures=0,
            # callbacks=tune_callbacks,
            # resume="AUTO"
        )

        logger.info("\n=== Training completed ===")
        run_successful = True

        # Анализируем результаты
        best_trial = analysis.get_best_trial(metric="episode_reward_mean", mode="max", scope="last")
        if best_trial:
            best_checkpoint_dict = analysis.get_best_checkpoint(trial=best_trial, metric="episode_reward_mean", mode="max")
            # --- ИСПРАВЛЕНИЕ AttributeError ---
            if best_checkpoint_dict:
                best_checkpoint_path = best_checkpoint_dict if isinstance(best_checkpoint_dict, str) else best_checkpoint_dict.get("filesystem", {}).get("path")
                logger.info(f"Best checkpoint found at: {best_checkpoint_path}")
                # TODO: Логирование лучшей модели в W&B
                # if use_wandb and best_checkpoint_path:
                #      hybrid_logger.log_model(best_checkpoint_path, "best_model_checkpoint")
            else:
                logger.warning("Could not find a best checkpoint for the best trial.")
            # --- Конец исправления ---
            logger.info(f"Best trial final results: {best_trial.last_result}")

        else:
             logger.warning("Could not determine the best trial based on 'episode_reward_mean'.")

        # Сохраняем PokerConfig (если есть метод save)
        # if hasattr(poker_config, 'save'): poker_config.save()


    except Exception as e:
        logger.error(f"\nCritical error during training: {e}", exc_info=True)

    finally:
        logger.info("\n=== Cleanup ===")
        # if use_wandb and wandb.run is not None:
        #      wandb.finish()
        ray.shutdown()
        logger.info("Ray shutdown completed.")
        if run_successful:
             logger.info("Training process finished successfully.")
        else:
             logger.error("Training process failed.")


if __name__ == "__main__":
    # Устанавливаем переменные окружения
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    train_poker()

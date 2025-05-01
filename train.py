# train.py
import os
import sys
import logging
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
# import wandb # Раскомментируй, если используешь W&B
# from ray.tune.logger import WandbLogger # Импорт для W&B в Ray 2.10
import numpy as np
from pathlib import Path
# --- ДОБАВЛЕН ИМПОРТ ---
# Для Ray 2.10 нужен PPO класс для default_resource_request
from ray.rllib.algorithms.ppo import PPO
# -----------------------

# Импортируем наши модули
from config import PokerConfig, TrainingConfig
from environment import PokerEnv
from models import AdvancedPokerModel # Убедись, что models.py есть

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
        training_config.model = poker_config.model_config

        # Настраиваем W&B (если нужно)
        use_wandb = setup_wandb(training_config)

        # --- Инициализируем Ray ---
        available_cpus = os.cpu_count() or 12
        # Устанавливаем num_workers = 5, как решили
        num_workers = 5
        logger.info(f"Setting num_workers to {num_workers}")

        ray.init(
            num_cpus=available_cpus, # Используем все CPU, что видит ОС
            num_gpus=training_config.num_gpus,
            logging_level=logging.INFO,
            ignore_reinit_error=True
        )
        logger.info(f"Ray sees AVAILABLE resources: {ray.available_resources()}")
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
                num_rollout_workers=num_workers, # Используем установленное значение
                num_envs_per_worker=training_config.num_envs_per_worker,
                rollout_fragment_length=training_config.rollout_fragment_length,
                batch_mode=training_config.batch_mode
            )
            .resources( # Ресурсы для Ray 2.10
                num_gpus=training_config.num_gpus, # GPU для learner'а
                num_cpus_per_worker=training_config.num_cpus_per_worker,
                num_gpus_per_worker=training_config.num_gpus_per_worker
                # УБИРАЕМ num_cpus_for_driver
            )
            # --- ВРЕМЕННО ОТКЛЮЧАЕМ EVALUATION ---
            # .evaluation(
            #     evaluation_interval=training_config.evaluation_interval,
            #     evaluation_duration=training_config.evaluation_duration,
            #     evaluation_num_workers=training_config.evaluation_num_workers,
            #     evaluation_parallel_to_training=False, # Отключаем параллельность
            #     evaluation_config={"explore": False}
            # )
            .debugging(log_level=training_config.log_level)
            # .callbacks(PokerCallbacks) # Раскомментируй, если нужны
        )
        final_ppo_config_dict = ppo_config_builder.to_dict()

        # --- ПЕЧАТАЕМ РАССЧИТАННЫЙ ЗАПРОС РЕСУРСОВ ---
        try:
             # Для Ray 2.10 используем PPO класс напрямую
             calculated_resources = PPO.default_resource_request(final_ppo_config_dict)
             logger.info(f"RLlib CALCULATED resource request per trial: {calculated_resources}")
        except Exception as e:
             logger.warning(f"Could not calculate default_resource_request: {e}")
        # ------------------------------------------
        logger.info("PPOConfig configured.")


        # --- Запускаем обучение с Ray Tune (classic API) ---
        logger.info(f"Starting Ray Tune experiment '{training_config.tune_exp_name}'...")
        local_dir_path = Path(training_config.local_dir)
        storage_parent_path = str(local_dir_path.parent.resolve())
        exp_dir_name = local_dir_path.name

        logger.info(f"Results will be stored under: {storage_parent_path}/{exp_dir_name}")
        # Tune сам создаст папку эксперимента, mkdir не нужен

        tune_callbacks = []
        # ... (W&B callback, если нужен) ...

        analysis = tune.run(
            "PPO",
            name=exp_dir_name,
            config=final_ppo_config_dict,
            stop={"training_iteration": training_config.num_iterations},
            local_dir=storage_parent_path,
            checkpoint_freq=training_config.checkpoint_freq,
            checkpoint_at_end=training_config.checkpoint_at_end,
            keep_checkpoints_num=training_config.keep_checkpoints_num,
            verbose=1,
            # fail_fast=True,
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
            if best_checkpoint_dict:
                best_checkpoint_path = best_checkpoint_dict if isinstance(best_checkpoint_dict, str) else best_checkpoint_dict.get("path")
                logger.info(f"Best checkpoint found at: {best_checkpoint_path}")
                # TODO: Логирование
            else:
                logger.warning("Could not find a best checkpoint.")
            logger.info(f"Best trial final results: {best_trial.last_result}")
        else:
             logger.warning("Could not determine the best trial.")

        # if hasattr(poker_config, 'save'): poker_config.save()


    except Exception as e:
        logger.error(f"\nCritical error during training: {e}", exc_info=True)

    finally:
        logger.info("\n=== Cleanup ===")
        # if use_wandb and wandb.run is not None: wandb.finish()
        ray.shutdown()
        logger.info("Ray shutdown completed.")
        if run_successful: logger.info("Training process finished successfully.")
        else: logger.error("Training process failed.")


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    train_poker()

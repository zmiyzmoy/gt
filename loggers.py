# loggers.py
import logging
import wandb
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class HybridLogger:
    """Логгер для Weights & Biases (без MLflow)."""
    def __init__(self, config): # Принимает TrainingConfig
        self.config = config
        self.wandb_run = None
        self._setup_wandb()

    def _setup_wandb(self):
        """Настройка W&B."""
        if self.config.wandb_project:
            try:
                api_key = os.environ.get("WANDB_API_KEY")
                if not api_key and os.environ.get("WANDB_MODE") != "online":
                     logger.warning("No WANDB_API_KEY. Running W&B in offline mode.")
                     os.environ["WANDB_MODE"] = "offline"
                elif api_key and os.environ.get("WANDB_MODE") != "offline":
                     os.environ["WANDB_MODE"] = "online" # Включаем online, если есть ключ

                self.wandb_run = wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.exp_name,
                    config=self.config.__dict__,
                    reinit=True
                )
                logger.info(f"W&B initialized (mode: {wandb.run.settings.mode if wandb.run else 'disabled'})")
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}. Disabling.")
                os.environ["WANDB_MODE"] = "disabled"
                self.wandb_run = None
        else:
             logger.info("W&B project not specified, skipping.")

    def log_metrics(self, metrics, step=None):
        """Логирует метрики в W&B."""
        if self.wandb_run and wandb.run:
            try: wandb.log(metrics, step=step)
            except Exception as e: logger.warning(f"W&B log_metrics failed: {e}")
        # logger.info(f"Metrics (step {step}): {metrics}") # Можно раскомментировать

    def log_model(self, checkpoint_path, artifact_name="model_checkpoint"):
        """Логирует чекпоинт модели как артефакт W&B."""
        if self.wandb_run and wandb.run:
            try:
                path_str = str(getattr(checkpoint_path, 'path', checkpoint_path)) # Получаем путь
                if not os.path.exists(path_str):
                     logger.warning(f"Checkpoint path not found: {path_str}")
                     return

                artifact_unique_name = f"{self.config.exp_name}-{artifact_name}"
                artifact = wandb.Artifact(name=artifact_unique_name, type='model')

                if os.path.isdir(path_str): artifact.add_dir(path_str)
                elif os.path.isfile(path_str): artifact.add_file(path_str)

                self.wandb_run.log_artifact(artifact)
                logger.info(f"Model artifact '{artifact.name}' logged to W&B.")
            except Exception as e:
                logger.warning(f"W&B log_model failed: {e}", exc_info=True)
        # logger.info(f"Model checkpoint processed: {checkpoint_path}")

    def end_run(self):
        """Завершает сессию W&B."""
        if self.wandb_run and wandb.run:
            try: wandb.finish()
            except Exception as e: logger.warning(f"W&B finish failed: {e}")
        logger.info("Logger finished.")

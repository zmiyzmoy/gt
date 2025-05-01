# loggers.py
import logging
import os # Добавим os для expanduser
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import wandb
    from ray.tune.logger import LoggerCallback
    from ray.tune.experiment.trial import Trial
    WANDB_AVAILABLE = True
except ImportError:
    logger.warning("wandb or ray not installed. WandbLoggerCallback will not be available.")
    WANDB_AVAILABLE = False

# Создаем фиктивный класс, если wandb не доступен
if not WANDB_AVAILABLE:
    class WandbLoggerCallback:
        def __init__(self, *args, **kwargs):
            logger.error("WandbLoggerCallback requires wandb and ray to be installed")
        def _init_wandb_run(self, trial):
            pass
        def log_trial_result(self, iteration, trial, result):
            pass
        def on_trial_complete(self, iteration, trials, trial, **info):
            pass
else:
    # --- ИСПРАВЛЕНО: Используем правильное имя и наследование ---
    class WandbLoggerCallback(LoggerCallback):
        """
        Коллбэк для логирования результатов Ray Tune в Weights & Biases.
        """
        # Константы для исключения стандартных ключей RLlib/Tune
        DEFAULT_EXCLUDES = (
            "trial_id",
            "experiment_tag",
            "date",
            "timestamp",
            "pid",
            "hostname",
            "node_ip",
            "config",
            "time_total_s",
            "timesteps_total",
            "episodes_total",
            "iterations_since_restore",
            "timesteps_since_restore",
            "episodes_since_restore",
            "warmup_time",
            "experiment_id",
            "time_this_iter_s",
            "time_since_restore",
            "done", # Ключ завершения RLlib
            "should_checkpoint" # Ключ для чекпоинтов RLlib
        )

        def __init__(
            self,
            project: Optional[str] = None,
            group: Optional[str] = None,
            run_name: Optional[str] = None, # Можно задать имя запуска
            config_dict: Optional[Dict] = None, # Переименовано из config для ясности
            api_key: Optional[str] = None,
            api_key_file: Optional[str] = None,
            log_config: bool = True,
            excludes: List[str] = [], # Дополнительные ключи для исключения
            **wandb_init_kwargs # Дополнительные аргументы для wandb.init
        ):
            self._project = project
            self._group = group
            self._run_name = run_name
            self._config_dict = config_dict if config_dict else {}
            self._log_config = log_config
            self._excludes = list(self.DEFAULT_EXCLUDES) + excludes
            self._wandb_init_kwargs = wandb_init_kwargs
            self._trial_runs: Dict[str, 'wandb.sdk.wandb_run.Run'] = {} # Храним W&B runs для каждого trial

            # --- Обработка API ключа ---
            resolved_api_key = api_key
            if api_key_file:
                try:
                    api_key_path = os.path.expanduser(api_key_file)
                    if os.path.exists(api_key_path):
                        with open(api_key_path, "r") as f:
                            resolved_api_key = f.read().strip()
                    else:
                        logger.warning(f"W&B API key file not found: {api_key_path}")
                except Exception as e:
                    logger.error(f"Error reading W&B API key file {api_key_file}: {e}")

            # --- Попытка логина W&B (если ключ есть) ---
            # Это лучше делать один раз при инициализации Ray, если возможно,
            # но можно и здесь попробовать.
            try:
                if resolved_api_key:
                    wandb.login(key=resolved_api_key)
                elif os.environ.get("WANDB_API_KEY"):
                    logger.info("Using WANDB_API_KEY from environment variables.")
                    wandb.login() # Попробует использовать ключ из окружения
                else:
                    logger.info("No W&B API key provided or found in environment. Will attempt anonymous login or rely on netrc.")
            except Exception as e:
                logger.error(f"W&B login failed: {e}. W&B logging might not work.")


        def _init_wandb_run(self, trial: Trial):
            """Инициализирует W&B run для конкретного trial."""
            trial_id = trial.trial_id
            if trial_id in self._trial_runs and self._trial_runs[trial_id] is not None:
                # Run уже инициализирован для этого trial
                return

            # Используем имя trial или заданное run_name
            # Добавляем ID trial к имени, чтобы избежать конфликтов при перезапусках
            run_name = f"{self._run_name or trial.experiment_tag}_{trial.trial_id[:8]}"

            # Формируем конфиг для W&B
            config_to_log = {}
            if self._log_config:
                # Логируем параметры trial (из Tune)
                config_to_log.update(trial.config)
                # Добавляем/перезаписываем общим конфигом, если он был передан
                if isinstance(self._config_dict, dict):
                    config_to_log.update(self._config_dict)

            logger.info(f"Initializing W&B run for trial {trial_id} (Run name: {run_name})")
            try:
                run = wandb.init(
                    project=self._project,
                    group=self._group,
                    name=run_name,
                    config=config_to_log if config_to_log else None,
                    id=f"{trial.experiment_tag}_{trial_id}", # Уникальный ID для возобновления
                    resume="allow", # Разрешаем возобновление
                    reinit=True, # Важно для нескольких trial в одном процессе
                    **self._wandb_init_kwargs
                )
                self._trial_runs[trial_id] = run
                logger.info(f"W&B run for trial {trial_id} initialized: {run.url}")
            except Exception as e:
                logger.error(f"Failed to initialize W&B run for trial {trial_id}: {e}", exc_info=True)
                self._trial_runs[trial_id] = None # Помечаем, что инициализация не удалась


        def log_trial_result(self, iteration: int, trial: Trial, result: Dict):
            """Вызывается после каждой итерации обучения для trial."""
            trial_id = trial.trial_id
            # Инициализируем run, если это первая итерация для этого trial
            if trial_id not in self._trial_runs:
                self._init_wandb_run(trial)

            run = self._trial_runs.get(trial_id)
            if run is None: # Если инициализация не удалась или run был закрыт
                # logger.warning(f"W&B run not available for trial {trial_id}. Skipping logging result.")
                return

            # Шаг для логирования (обычно training_iteration)
            step = result.get("training_iteration", iteration)

            # Фильтруем и подготавливаем метрики
            log_dict = {}
            flat_result = self._flatten_dict(result) # Сглаживаем вложенные словари

            for key, value in flat_result.items():
                if key not in self._excludes:
                    # Конвертируем numpy типы и другие несериализуемые
                    log_value = self._try_make_serializable(value)
                    if log_value is not None: # Логируем, только если значение удалось преобразовать
                        log_dict[key] = log_value
                    # else:
                    #    logger.debug(f"Skipping non-serializable key '{key}' type {type(value)} for W&B")

            # Логируем метрики в W&B
            if log_dict:
                try:
                    run.log(log_dict, step=step)
                except Exception as e:
                    logger.error(f"Failed to log metrics to W&B for trial {trial_id} at step {step}: {e}")

        def on_trial_complete(self, iteration: int, trials: List[Trial], trial: Trial, **info):
            """Вызывается при завершении trial (успешном или с ошибкой)."""
            run = self._trial_runs.pop(trial.trial_id, None) # Удаляем и получаем run
            if run:
                logger.info(f"Finishing W&B run for trial {trial.trial_id} (status: {trial.status})")
                exit_code = 0 if trial.status == Trial.TERMINATED else 1
                try:
                    run.finish(exit_code=exit_code)
                except Exception as e:
                    logger.error(f"Error finishing W&B run for trial {trial.trial_id}: {e}")

        def _flatten_dict(self, d, parent_key='', sep='/'):
            """Сглаживает вложенный словарь."""
            items = []
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                if isinstance(v, dict):
                    items.extend(self._flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        def _try_make_serializable(self, value):
            """Пытается преобразовать значение в тип, поддерживаемый W&B."""
            if isinstance(value, (str, int, float, bool)):
                return value
            if hasattr(value, 'item'): # numpy скаляры
                return value.item()
            if isinstance(value, (list, tuple)) and all(isinstance(x, (str, int, float, bool)) for x in value):
                return value # Простые списки/кортежи
            # Добавить обработку других типов при необходимости (например, wandb.Image)
            return None # Возвращаем None, если не можем сериализовать

        def __del__(self):
            """Гарантирует закрытие всех оставшихся runs при удалении объекта."""
            for trial_id, run in list(self._trial_runs.items()):
                if run:
                    logger.warning(f"Finishing W&B run for trial {trial_id} during cleanup (likely due to error).")
                    try:
                        run.finish(exit_code=1, quiet=True) # Завершаем с кодом ошибки
                    except Exception as e:
                        logger.error(f"Error finishing W&B run during cleanup for {trial_id}: {e}")
                self._trial_runs.pop(trial_id, None)

# Можно добавить другие коллбэки или логгеры при необходимости
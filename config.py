# config.py
from pathlib import Path # Для работы с путями файловой системы
from datetime import datetime # Для генерации уникальных имен на основе времени
import json # Для форматирования game_config в JSON
import numpy as np # Для работы с массивами данных (если нужно)
from dataclasses import dataclass, field # Для создания конфигурационных классов
from typing import Dict, Any, List     # Для аннотаций типов

@dataclass
class PokerConfig:
    # --- Параметры Игры (OpenSpiel) ---
    game_name: str = "universal_poker"
    # --- Храним ПРАВИЛЬНЫЕ типы здесь (в виде списков/чисел) ---
    game_config: Dict[str, Any] = field(default_factory=dict)

    # --- Параметры Среды ---
    num_players: int = 8
    starting_stack: int = 10000 # Стек для одного игрока
    small_blind: int = 50
    big_blind: int = 100

    # --- Параметры Модели ---
    model_config: Dict[str, Any] = field(default_factory=lambda: {
        "custom_model": "AdvancedPokerModel", # Убедитесь, что модель с таким именем зарегистрирована в RLlib
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
        # Дополнительные параметры кастомной модели, если они нужны
        # "custom_model_config": {},
    })

    def __post_init__(self):
        # --- Заполнение game_config НУЖНЫМИ ТИПАМИ ---
        if not self.game_config:
            # Определяем количество карт на каждой улице (пример для Hold'em: префлоп(0), флоп(3), терн(1), ривер(1))
            board_cards = [0, 3, 1, 1] # Должно быть 4 раунда для Hold'em

            # Создаем список стартовых стеков, по одному значению для каждого игрока
            stacks = [self.starting_stack] * self.num_players

            # --- ИСПРАВЛЕНИЕ для БЛАЙНДОВ ---
            # Создаем список блайндов для КАЖДОГО игрока
            # Стандартная структура для NLHE 8-max: SB, BB, остальные 0
            if self.num_players >= 2:
                blinds = [self.small_blind, self.big_blind] + [0] * (self.num_players - 2)
            elif self.num_players == 1: # Маловероятный случай
                blinds = [0]
            else: # num_players = 0?
                blinds = []

            # Дополнительная проверка на случай, если num_players < 2 и код выше дал некорректный результат
            if len(blinds) != self.num_players:
                 print(f"Warning: Blind list length mismatch! Expected {self.num_players}, got {len(blinds)}. Reconstructing blinds.")
                 # Попытка создать корректный список заново
                 if self.num_players >= 2:
                      blinds = [self.small_blind, self.big_blind] + [0] * (self.num_players - 2)
                 elif self.num_players == 1:
                      blinds = [0]
                 else:
                      blinds = []
                 # Если все еще не совпадает, возможно, num_players == 0 или другая проблема
                 if len(blinds) != self.num_players:
                      print(f"Error: Could not create correct blind list for {self.num_players} players.")
                      # Можно либо выбросить ошибку, либо установить по умолчанию, но это может не сработать
                      # raise ValueError(f"Cannot configure blinds for {self.num_players} players.")
                      blinds = [0] * self.num_players # Заполняем нулями как запасной вариант

            # --- КОНЕЦ ИСПРАВЛЕНИЯ для БЛАЙНДОВ ---

            self.game_config = {
                # Параметры, ожидаемые universal_poker (см. документацию OpenSpiel или .game файлы)
                "betting": "nolimit",               # str: тип ставок (limit, potlimit, nolimit)
                "numPlayers": self.num_players,     # int: количество игроков
                "numRounds": len(board_cards),      # int: количество раундов торговли (улиц)
                "numSuits": 4,                      # int: количество мастей
                "numRanks": 13,                     # int: количество рангов карт
                "numHoleCards": 2,                  # int: количество карманных карт
                "numBoardCards": board_cards,       # list[int]: количество общих карт на каждой улице
                "stack": stacks,                    # list[int]: список стартовых стеков
                "blind": blinds,                    # list[int]: список блайндов для КАЖДОГО игрока

                # --- Опциональные параметры (можно добавить при необходимости) ---
                # "firstPlayer": [1, 0, 0, 0],       # list[int]: Позиция первого игрока на каждой улице (1=SB, 0=BB...) - зависит от правил
                # "maxRaises": [4, 4, 4, 4],         # list[int]: Макс. кол-во рейзов (нужно для limit)
                # "betRaises": [100, 200, 200, 200]  # list[int]: Размеры ставок/рейзов (нужно для limit)
                # "haStackSize": "0"                 # str: (Для Heads-Up) - здесь не нужно
            }
        # Преобразование списков в строки будет происходить в environment.py перед вызовом load_game

        # Устанавливаем base_dir для сохранения результатов/логов
        # Используем относительный путь от места запуска скрипта
        self.base_dir = Path(f"./poker_prod_{datetime.now().strftime('%Y%m%d_%H%M%S')}")


@dataclass
class TrainingConfig:
    # --- Общие параметры Эксперимента ---
    exp_name: str = f"poker_prod_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    local_dir: str = "./ray_results" # Каталог для сохранения результатов Ray Tune/RLlib

    # --- Параметры Обучения (для Ray Tune) ---
    num_iterations: int = 1000       # Общее количество итераций обучения
    checkpoint_freq: int = 25        # Сохранять чекпоинт каждые N итераций
    keep_checkpoints_num: int = 3    # Сколько последних чекпоинтов хранить
    checkpoint_at_end: bool = True   # Сохранять ли чекпоинт в конце обучения

    # --- Параметры Логирования ---
    log_level: str = "INFO"         # Уровень логирования (DEBUG, INFO, WARN, ERROR)
    wandb_project: str = "poker_rl" # Название проекта в Weights & Biases (если используется)
    # wandb_api_key_file: str = "~/.wandb_api_key" # Путь к файлу с ключом W&B (если не задан глобально)

    # --- Параметры Алгоритма PPO (для Ray RLlib PPOConfig v2.10+) ---
    # Ресурсы
    num_workers: int = 7           # Количество rollout worker'ов (часто num_cpu - 1)
    num_gpus: int = 1              # Количество GPU для основного процесса обучения (тренера)
    num_cpus_per_worker: int = 1   # Количество CPU на каждый rollout worker
    num_gpus_per_worker: float = 0.0 # Количество GPU на каждый rollout worker (обычно 0)
    num_envs_per_worker: int = 1   # Количество экземпляров среды на одном worker'е

    # Параметры Обучения PPO
    lr: float = 5e-5               # Скорость обучения (learning rate)
    gamma: float = 0.99            # Фактор дисконтирования
    lambda_: float = 0.95          # Параметр для GAE (Generalized Advantage Estimation)
    clip_param: float = 0.2        # Параметр клиппинга для PPO loss
    vf_loss_coeff: float = 0.5     # Коэффициент для Value Function loss
    entropy_coeff: float = 0.01    # Коэффициент для энтропийного бонуса (регуляризация)
    train_batch_size: int = 8192   # Общий размер батча для одной итерации SGD (со всех воркеров)
    sgd_minibatch_size: int = 1024 # Размер мини-батча для одной SGD эпохи
    num_sgd_iter: int = 10         # Количество SGD эпох за одну итерацию обучения

    # Параметры Сбора Данных (Rollout)
    rollout_fragment_length: str = "auto" # Длина фрагмента траектории, собираемого одним воркером за раз
    batch_mode: str = "truncate_episodes" # Режим формирования батчей

    # Параметры Оценки (Evaluation)
    evaluation_interval: int = 20           # Запускать оценку каждые N итераций обучения
    evaluation_duration: int = 10           # Количество эпизодов для оценки
    evaluation_num_workers: int = 1         # Количество воркеров для оценки (может быть 0, если оценка на тренере)
    evaluation_parallel_to_training: bool = False # Запускать ли оценку параллельно с обучением
    evaluation_config: Dict[str, Any] = field(default_factory=dict) # Дополнительные параметры для оценки

    # Параметры Модели (передаются в PPOConfig)
    model: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Если модель не задана явно, берем из PokerConfig по умолчанию
        if not self.model:
             poker_cfg = PokerConfig() # Создаем экземпляр для доступа к model_config
             self.model = poker_cfg.model_config

        # Устанавливаем имя эксперимента для Ray Tune
        self.tune_exp_name = self.exp_name

        # Можно добавить базовые параметры оценки, если они не заданы
        if not self.evaluation_config:
            self.evaluation_config = {
                "explore": False, # Обычно отключают exploration во время оценки
                # Можно переопределить env_config для оценки, если нужно
                # "env_config": {...},
            }


# Пример использования (для проверки при запуске файла напрямую)
if __name__ == "__main__":
    poker_config = PokerConfig()
    train_config = TrainingConfig()

    print("--- Poker Config ---")
    print(f"Game Name: {poker_config.game_name}")
    print(f"Num Players: {poker_config.num_players}")
    print(f"Starting Stack: {poker_config.starting_stack}")
    print(f"Small Blind: {poker_config.small_blind}")
    print(f"Big Blind: {poker_config.big_blind}")
    print("\n--- Generated game_config (for OpenSpiel) ---")
    # Выводим в формате JSON для наглядности
    print(json.dumps(poker_config.game_config, indent=2))

    print("\n--- Training Config ---")
    print(f"Experiment Name: {train_config.exp_name}")
    print(f"Num Workers: {train_config.num_workers}")
    print(f"Learning Rate: {train_config.lr}")
    print(f"Train Batch Size: {train_config.train_batch_size}")
    print(f"Model Config: {train_config.model}") # Модель должна взяться из PokerConfig
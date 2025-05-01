# config.py
from pathlib import Path
from datetime import datetime
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class PokerConfig:
    # --- Параметры Игры (OpenSpiel) ---
    game_name: str = "universal_poker"
    # --- Храним ПРАВИЛЬНЫЕ типы здесь ---
    game_config: Dict[str, Any] = field(default_factory=dict)

    # --- Параметры Среды ---
    num_players: int = 8
    starting_stack: int = 10000 # Стек для одного игрока
    small_blind: int = 50
    big_blind: int = 100

    # --- Параметры Модели ---
    model_config: Dict[str, Any] = field(default_factory=lambda: {
        "custom_model": "AdvancedPokerModel",
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
    })

    def __post_init__(self):
        # --- Заполнение game_config НУЖНЫМИ ТИПАМИ ---
        if not self.game_config:
            blinds = [self.small_blind, self.big_blind]
            # Определяем количество карт на каждой улице (пример для Hold'em: префлоп(0), флоп(3), терн(1), ривер(1))
            board_cards = [0, 3, 1, 1]

            # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
            # Создаем список стартовых стеков, по одному значению для каждого игрока
            stacks = [self.starting_stack] * self.num_players # -> [10000, 10000, ..., 10000] (8 раз)
            # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

            self.game_config = {
                "betting": "nolimit",               # str: тип ставок (например, limit, potlimit, nolimit)
                "numPlayers": self.num_players,     # int: количество игроков
                "numRounds": len(board_cards),      # int: количество раундов торговли (улиц)
                "numSuits": 4,                      # int: количество мастей
                "numRanks": 13,                     # int: количество рангов карт
                "numHoleCards": 2,                  # int: количество карманных карт
                "numBoardCards": board_cards,       # list[int]: количество общих карт, открываемых на каждой улице
                # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
                "stack": stacks,                    # list[int]: список стартовых стеков для каждого игрока
                # --- КОНЕЦ ИСПРАВЛЕНИЯ ---
                "blind": blinds                     # list[int]: размеры блайндов (малый, большой, возможно, другие)
                # Дополнительные параметры universal_poker (если нужны):
                # "maxRaises": [...], # Макс. кол-во рейзов на каждой улице (для limit)
                # "betRaises": [...], # Размеры ставок/рейзов на каждой улице (для limit)
                # "firstPlayer": [...] # Позиции первого ходящего на каждой улице
                # ... и другие ...
            }
        # Конвертация в строки/нужные типы будет в environment.py перед load_game

        # Устанавливаем base_dir
        # Используем относительный путь от места запуска скрипта
        self.base_dir = Path(f"./poker_prod_{datetime.now().strftime('%Y%m%d_%H%M%S')}")


@dataclass
class TrainingConfig:
    # --- Общие параметры Эксперимента ---
    exp_name: str = f"poker_prod_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    local_dir: str = "./ray_results" # Используем local_dir для Ray 2.10

    # --- Параметры Обучения (для Ray Tune) ---
    num_iterations: int = 1000
    checkpoint_freq: int = 25
    keep_checkpoints_num: int = 3
    checkpoint_at_end: bool = True

    # --- Параметры Логирования ---
    log_level: str = "INFO"
    wandb_project: str = "poker_rl" # Оставляем W&B, убрали MLflow

    # --- Параметры Алгоритма PPO (для Ray RLlib PPOConfig v2.10) ---
    # Ресурсы
    num_workers: int = 6 # Оставляем 6 для 8 доступных CPU
    num_gpus: int = 1
    num_cpus_per_worker: int = 1
    num_gpus_per_worker: float = 0.0
    num_envs_per_worker: int = 1
    # Обучение
    lr: float = 5e-5
    gamma: float = 0.99
    lambda_: float = 0.95
    clip_param: float = 0.2
    vf_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    train_batch_size: int = 8192
    sgd_minibatch_size: int = 1024
    num_sgd_iter: int = 10
    # Rollout
    rollout_fragment_length: str = "auto"
    batch_mode: str = "truncate_episodes"
    # Evaluation
    evaluation_interval: int = 20
    evaluation_duration: int = 10
    evaluation_num_workers: int = 1 # Отдельный воркер для оценки
    evaluation_parallel_to_training: bool = False # Отключили для экономии CPU
    # Модель
    model: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.model:
             poker_cfg = PokerConfig() # Создаем экземпляр для доступа к model_config по умолчанию
             self.model = poker_cfg.model_config
        # Используем exp_name для имени эксперимента в Tune
        self.tune_exp_name = self.exp_name

# Пример использования (не обязательно для работы скрипта)
if __name__ == "__main__":
    poker_config = PokerConfig()
    train_config = TrainingConfig()

    print("--- Poker Config ---")
    print(poker_config)
    print("\n--- Training Config ---")
    print(train_config)

    # Проверка game_config после __post_init__
    print("\n--- Generated game_config ---")

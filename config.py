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
    # --- ИЗМЕНЕНО: Тип Any, чтобы хранить разные типы ---
    game_config: Dict[str, Any] = field(default_factory=dict)

    # --- Параметры Среды ---
    num_players: int = 8
    starting_stack: int = 10000
    small_blind: int = 50
    big_blind: int = 100

    # --- Параметры Модели ---
    model_config: Dict[str, Any] = field(default_factory=lambda: {
        "custom_model": "AdvancedPokerModel",
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
    })

    def __post_init__(self):
        # --- ИЗМЕНЕНО: Заполняем правильными типами, конвертация убрана отсюда ---
        if not self.game_config:
            self.game_config = {
                "betting": "nolimit",               # str
                "numPlayers": self.num_players,     # int
                "numRounds": 4,                     # int
                "numSuits": 4,                      # int
                "numRanks": 13,                     # int
                "numHoleCards": 2,                  # int
                "numBoardCards": [0, 3, 1, 1],      # list[int]
                "stack": self.starting_stack,       # int
                "blind": [self.small_blind, self.big_blind] # list[int]
            }

        # --- Параметры ниже не относятся к game_config ---
        # Устанавливаем base_dir
        self.base_dir = Path(f"./poker_prod_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

@dataclass
class TrainingConfig:
    # --- Общие ---
    exp_name: str = f"poker_prod_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    local_dir: str = "./ray_results" # Для Ray 2.10

    # --- Обучение (Tune) ---
    num_iterations: int = 1000
    checkpoint_freq: int = 25
    keep_checkpoints_num: int = 3
    checkpoint_at_end: bool = True

    # --- Логирование ---
    log_level: str = "INFO"
    wandb_project: str = "poker_rl"

    # --- PPO (RLlib 2.10) ---
    # Ресурсы
    num_workers: int = 6 # Уменьшено для 8 CPU
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
    evaluation_num_workers: int = 1
    evaluation_parallel_to_training: bool = False # Отключили для экономии CPU
    # Модель
    model: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.model:
             poker_cfg = PokerConfig()
             self.model = poker_cfg.model_config
        self.tune_exp_name = self.exp_name # Используем одно имя для простоты

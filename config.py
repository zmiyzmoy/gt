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
    game_config: Dict[str, str] = field(default_factory=dict)

    # --- Параметры Среды ---
    num_players: int = 8
    starting_stack: int = 10000
    small_blind: int = 50
    big_blind: int = 100

    # --- Параметры Модели (для RLlib PPOConfig) ---
    model_config: Dict[str, Any] = field(default_factory=lambda: {
        "custom_model": "AdvancedPokerModel",
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
    })

    def __post_init__(self):
        if not self.game_config:
            self.game_config = {
                "betting": "nolimit",
                "numPlayers": str(self.num_players),
                "numRounds": "4",
                "numSuits": "4",
                "numRanks": "13",
                "numHoleCards": "2",
                "numBoardCards": "0 3 1 1",
                "stack": str(self.starting_stack),
                "blind": f"{self.small_blind} {self.big_blind}"
            }
        temp_config = {}
        for k, v in self.game_config.items():
             temp_config[k] = str(v)
        self.game_config = temp_config


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
    wandb_project: str = "poker_rl"

    # --- Параметры Алгоритма PPO (для Ray RLlib PPOConfig v2.10) ---
    # Ресурсы
    num_workers: int = 6 # Оставляем 6
    num_gpus: int = 1
    num_cpus_per_worker: int = 1
    num_gpus_per_worker: float = 0.0
    num_envs_per_worker: int = 1

    # Параметры обучения
    lr: float = 5e-5
    gamma: float = 0.99
    lambda_: float = 0.95
    clip_param: float = 0.2
    vf_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    train_batch_size: int = 8192
    sgd_minibatch_size: int = 1024
    num_sgd_iter: int = 10

    # Параметры Rollout
    rollout_fragment_length: str = "auto"
    batch_mode: str = "truncate_episodes"

    # Параметры Evaluation
    evaluation_interval: int = 20
    evaluation_duration: int = 10
    evaluation_num_workers: int = 1
    evaluation_parallel_to_training: bool = False # <--- ИЗМЕНЕНО ЗДЕСЬ!

    # Модель (ссылка на PokerConfig.model_config)
    model: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.model:
             poker_cfg = PokerConfig()
             self.model = poker_cfg.model_config
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.tune_exp_name = f"{self.exp_name}_{timestamp}"

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
    # --- ИЗМЕНЕНО: Тип Any для хранения разных типов ---
    game_config: Dict[str, Any] = field(default_factory=dict)

    # --- Параметры Среды ---
    num_players: int = 8
    starting_stack: int = 10000 # Стартовый стек для ОДНОГО игрока
    small_blind: int = 50
    big_blind: int = 100

    # --- Параметры Модели ---
    model_config: Dict[str, Any] = field(default_factory=lambda: {
        "custom_model": "AdvancedPokerModel",
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
    })

    def __post_init__(self):
        # --- Заполнение game_config ПРАВИЛЬНЫМИ типами ---
        if not self.game_config:
            # Списки храним как списки, числа как числа
            blinds = [self.small_blind, self.big_blind]
            board_cards = [0, 3, 1, 1]

            self.game_config = {
                "betting": "nolimit",               # str
                "numPlayers": self.num_players,     # int
                "numRounds": 4,                     # int
                "numSuits": 4,                      # int
                "numRanks": 13,                     # int
                "numHoleCards": 2,                  # int
                "numBoardCards": board_cards,       # list[int]
                "stack": self.starting_stack,       # int <--- Оставляем как INT
                "blind": blinds                     # list[int]
            }

        self.base_dir = Path(f"./poker_prod_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

@dataclass
class TrainingConfig:
    # --- Общие параметры Эксперимента ---
    exp_name: str = f"poker_prod_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    local_dir: str = "./ray_results"

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
    num_workers: int = 6
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
    num__spiel.python import rl_environment
import pyspiel
from collections import deque
import logging
from typing import Dict, Any, List, Tuple

# Настройка логирования
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # Раскомментируй для детальной отладки

class OpponentStats:
    # ... (код OpponentStats остается БЕЗ ИЗМЕНЕНИЙ) ...
    def __init__(self, num_players): self.num_players = num_players; self.stats = {}; self.feature_size = 4; self.reset()
    def reset(self): self.stats = {}; [self._ensure_player_stats(i) for i in range(self.num_players)]
    def _ensure_player_stats(self, i): i=int(i); self.stats.setdefault(i,{'vpip':0,'pfr':0,'af':1,'hands':0,'agg':0,'pass':0,'vpip_opp':0,'pfr_opp':0,'vpip_act_count':0,'pfr_act_count':0,'vpip_acted_this_hand':False,'pfr_acted_this_hand':False}) # Добавил _count
    def get_features(self, i): i=int(i); self._ensure_player_stats(i); s=self.stats[i]; h=s.get('hands',0); vo=s.get('vpip_opportunities',0); po=s.get('pfr_opportunities',0); va=s.get('vpip_actions',0); pa=s.get('pfr_actions',0); ag=s.get('agg_actions',0); ps=s.get('pass_actions',0); vpip=float(va/vo)if vo>0 else 0; pfr=float(pa/po)if po>0 else 0; af=float(ag/ps)if ps>0 else(5 if ag>0 else 1); naf=min(af/5,1); nh=min(h/100,1); return np.array([vpip,pfr,naf,nh],dtype=np.float32)
    def record_opportunity(self, i, v, p): i=int(i); self._ensure_player_stats(i); if v: self.stats[i]['vpip_opportunities']=self.stats[i].get('vpip_opportunities',0)+1; self.stats[i]['vpip_acted_this_hand']=False; if p: self.stats[i]['pfr_opportunities']=self.stats[i].get('pfr_opportunities',0)+1; self.stats[i]['pfr_acted_this_hand']=False
    def update_on_action(self, i, t, s, vol): i=int(i); self._ensure_player_stats(i); st=self.stats[i]; if s==0 and vol and not st['vpip_acted_this_hand']: st['vpip_actions']=st.get('vpip_actions',0)+1; st['vpip_acted_this_hand']=True; if s==0 and t=='raise' and not st['pfr_acted_this_hand']: st['pfr_actions']=st.get('pfr_actions',0)+1; st['pfr_acted_this_hand']=True; if s>0: (st['agg_actions']:=st.get('agg_actions',0)+1) if t=='raise' else (st['pass_actions']:=st.get('pass_actions',0)+1) if t=='call' else None # Исправлено :=
    def finalize_hand_stats(self, inv): [ (self._ensure_player_stats(int(p)), self.stats[int(p)].update({'hands': self.stats[int(p)].get('hands',0)+1, 'vpip_acted_this_hand': False, 'pfr_acted_this_hand': False})) for p in inv]


class PokerEnv(gym.Env):
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}

    def __init__(self, env_config: dict):
        super().__init__()
        logger.info("Initializing PokerEnv...")

        if "config" not in env_config:
             raise ValueError("Missing 'config' key in env_config dictionary.")
        self.config = env_config["config"] # Экземпляр PokerConfig
        self.dtype = env_config.get("dtype", np.float32)

        # --- OpenSpiel Initialization ---
        # --- ИЗМЕНЕНО: ПРАВИЛЬНАЯ конвертация для OpenSpiel 1.3 ---
        game_params_from_config = self.config.game_config
        processed_game_params = {}
        for k, v in game_params_from_config.items():
            # Конвертируем списки в строки с пробелами
            if k in ["numBoardCards", "blind", "stack"]:
                if isinstance(v, (list, tuple, np.ndarray)):
                    processed_game_params[k] = " ".join(map(str, v))
                else: # Если вдруг пришло не списком, конвертируем в строку
                    processed_game_params[k] = str(v)
            # Оставляем числа числами (int)
            elif k in ["numPlayers", "numRounds", "numSuits", "numRanks",sgd_iter: int = 10
    # Rollout
    rollout_fragment_length: str = "auto"
    batch_mode: str = "truncate_episodes"
    # Evaluation
    evaluation_interval: int = 20
    evaluation_duration: int = 10
    evaluation_num_workers: int = 1
    evaluation_parallel_to_training: bool = False
    # Модель
    model: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.model:
             poker_cfg = PokerConfig()
             self.model = poker_cfg.model_config
        self.tune_exp_name = self.exp_name

import os
import ray
import pyspiel
import numpy as np
import torch
import torch.nn as nn
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from open_spiel.python import rl_environment, policy
from sklearn.cluster import KMeans
import joblib
from typing import Dict, Any, List, Tuple, Optional
import random
import logging
import json
import time
import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from collections import deque
import warnings
import gc
import sys

# ======================
# Расширенная конфигурация
# ======================
class PokerConfig:
    def __init__(self):
        # Базовые параметры игры
        self.num_players = 8
        self.game_name = (
            "universal_poker(betting=nolimit,numPlayers=8,numRounds=4,"
            "blind=1 2 3 4 5 6 7 8,raiseSize=0.10 0.20 0.40 0.80,"
            "stack=1000 1000 1000 1000 1000 1000 1000 1000,"
            "numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1)"
        )

        # Директории и пути
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(f"./poker_production_{self.timestamp}")
        self.model_dir = self.base_dir / "models"
        self.log_dir = self.base_dir / "logs"
        self.data_dir = self.base_dir / "data"
        self.kmeans_path = self.model_dir / "kmeans.joblib"
        self.stats_path = self.data_dir / "opponent_stats.json"
        self.metrics_path = self.data_dir / "metrics.csv"

        # Параметры обучения
        self.batch_size = 16384
        self.rollout_fragment_length = 160
        self.num_workers = 10
        self.num_envs_per_worker = 3
        self.sgd_minibatch_size = 4096
        self.num_sgd_iter = 5
        self.lr = 5e-5
        self.gamma = 0.97
        self.gpu_fraction = 0.95

        # Параметры для продакшена
        self.checkpoint_freq = 30
        self.eval_freq = 1000
        self.eval_episodes = 1000
        self.max_memory_gb = 80
        self.max_training_steps = 100_000_000
        self.early_stopping_patience = 10
        self.min_improvement = 0.01

        # Параметры для эксплойта
        self.opponent_types = ['TAG', 'LAG', 'Fish', 'Unknown']
        self.min_hands_for_stats = 100
        self.stats_update_freq = 50
        self.exploit_threshold = 0.1  # bb/100 hands

        self._create_directories()
        self._setup_logging()
        self._validate_config()

    def _create_directories(self):
        """Создание необходимых директорий"""
        for dir_path in [self.model_dir, self.log_dir, self.data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'training.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('poker_training')

    def _validate_config(self):
        """Валидация конфигурации"""
        assert self.num_players > 1, "Требуется минимум 2 игрока"
        assert self.batch_size > 0, "Batch size должен быть положительным"
        assert 0 < self.gamma <= 1, "Gamma должна быть в диапазоне (0,1]"
        assert self.max_memory_gb > 0, "Максимальная память должна быть положительной"

        # Проверка доступной памяти
        available_memory = psutil.virtual_memory().available / (1024**3)
        if available_memory < self.max_memory_gb:
            warnings.warn(f"Доступно только {available_memory:.1f}GB памяти")

    def save(self):
        """Сохранение конфигурации"""
        config_dict = {k: v for k, v in self.__dict__.items()
                      if not k.startswith('_') and isinstance(v, (int, float, str, bool, list, dict))}
        with open(self.base_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=4)

    @classmethod
    def load(cls, path):
        """Загрузка конфигурации"""
        with open(path) as f:
            config_dict = json.load(f)
        config = cls()
        for k, v in config_dict.items():
            setattr(config, k, v)
        return config

# Глобальная конфигурация
config = PokerConfig()
writer = SummaryWriter(log_dir=config.log_dir)

# ======================
# Улучшенная статистика оппонентов
# ======================
class OpponentStats:
    def __init__(self):
        self.stats = {i: {
            # Базовые метрики
            'vpip': 0, 'pfr': 0, 'af': 0, 'hands': 0,
            'fold_to_cbet': 0, 'cbet_opp': 0,
            'fold_to_3bet': 0, '3bet_opp': 0,
            'call_freq': 0, 'raise_freq': 0,

            # Расширенные метрики
            'street_aggression': [0] * 4,
            'position_stats': {pos: {'wins': 0, 'hands': 0, 'vpip': 0, 'pfr': 0}
                             for pos in range(8)},
            'vs_raise_stats': {'fold': 0, 'call': 0, 'raise': 0, 'total': 0},
            'vs_3bet_stats': {'fold': 0, 'call': 0, '4bet': 0, 'total': 0},

            # Метрики по размерам ставок
            'bet_sizing': {
                'preflop': deque(maxlen=100),
                'flop': deque(maxlen=100),
                'turn': deque(maxlen=100),
                'river': deque(maxlen=100)
            },

            # История рук для анализа
            'recent_hands': deque(maxlen=1000),

            # Временные метрики
            'last_update': time.time(),
            'total_winnings': 0,
            'winnings_by_position': [0] * 8
        } for i in range(config.num_players)}

        self.load_stats()

    def update(self, player_id, action, stage, pot, bet_size, position,
               vs_raise=False, vs_3bet=False, won=0, hand_info=None):
        """Обновление статистики с расширенными метриками"""
        stats = self.stats[player_id]
        stats['hands'] += 1
        stats['position_stats'][position]['hands'] += 1

        # Обновление базовых метрик
        if stage == 0:  # префлоп
            if action > 0:
                stats['vpip'] += 1
                stats['position_stats'][position]['vpip'] += 1
            if action > 1:
                stats['pfr'] += 1
                stats['position_stats'][position]['pfr'] += 1

        # Обновление статистики против рейза/3бета
        if vs_raise:
            stats['vs_raise_stats']['total'] += 1
            if action == 0:
                stats['vs_raise_stats']['fold'] += 1
            elif action == 1:
                stats['vs_raise_stats']['call'] += 1
            else:
                stats['vs_raise_stats']['raise'] += 1

        if vs_3bet:
            stats['vs_3bet_stats']['total'] += 1
            if action == 0:
                stats['vs_3bet_stats']['fold'] += 1
            elif action == 1:
                stats['vs_3bet_stats']['call'] += 1
            else:
                stats['vs_3bet_stats']['4bet'] += 1

        # Обновление размеров ставок
        if action > 1 and bet_size > 0:
            if stage == 0:
                stats['bet_sizing']['preflop'].append(bet_size / pot)
            elif stage == 1:
                stats['bet_sizing']['flop'].append(bet_size / pot)
            elif stage == 2:
                stats['bet_sizing']['turn'].append(bet_size / pot)
            else:
                stats['bet_sizing']['river'].append(bet_size / pot)

        # Обновление выигрышей
        if won != 0:
            stats['total_winnings'] += won
            stats['winnings_by_position'][position] += won

        # Сохранение информации о руке
        if hand_info:
            stats['recent_hands'].append(hand_info)

        stats['last_update'] = time.time()

        # Периодическое сохранение
        if stats['hands'] % config.stats_update_freq == 0:
            self.save_stats()

    def get_features(self, player_id):
        """Получение расширенных признаков для модели"""
        stats = self.stats[player_id]
        hands = max(1, stats['hands'])

        # Базовые признаки
        basic_features = [
            stats['vpip'] / hands,
            stats['pfr'] / hands,
            stats['af'] / hands,
            stats['fold_to_cbet'] / max(1, stats['cbet_opp']),
            stats['fold_to_3bet'] / max(1, stats['3bet_opp']),
            stats['call_freq'] / hands,
            stats['raise_freq'] / hands
        ]

        # Признаки по позициям
        position_features = []
        for pos in range(8):
            pos_stats = stats['position_stats'][pos]
            pos_hands = max(1, pos_stats['hands'])
            position_features.extend([
                pos_stats['vpip'] / pos_hands,
                pos_stats['pfr'] / pos_hands,
                pos_stats['wins'] / pos_hands
            ])

        # Признаки по размерам ставок
        sizing_features = []
        for street in ['preflop', 'flop', 'turn', 'river']:
            sizes = stats['bet_sizing'][street]
            sizing_features.extend([
                np.mean(sizes) if sizes else 0,
                np.std(sizes) if len(sizes) > 1 else 0
            ])

        # Признаки по реакции на агрессию
        vs_raise = stats['vs_raise_stats']
        vs_3bet = stats['vs_3bet_stats']
        aggression_features = [
            vs_raise['fold'] / max(1, vs_raise['total']),
            vs_raise['raise'] / max(1, vs_raise['total']),
            vs_3bet['fold'] / max(1, vs_3bet['total']),
            vs_3bet['4bet'] / max(1, vs_3bet['total'])
        ]

        return np.array(basic_features + position_features +
                       sizing_features + aggression_features)

    def save_stats(self):
        """Сохранение статистики в файл"""
        try:
            stats_dict = {
                player_id: {
                    k: (list(v) if isinstance(v, deque) else v)
                    for k, v in player_stats.items()
                    if k != 'recent_hands'
                }
                for player_id, player_stats in self.stats.items()
            }
            with open(config.stats_path, 'w') as f:
                json.dump(stats_dict, f)
        except Exception as e:
            logging.error(f"Ошибка при сохранении статистики: {e}")

    def load_stats(self):
        """Загрузка статистики из файла"""
        if config.stats_path.exists():
            try:
                with open(config.stats_path) as f:
                    stats_dict = json.load(f)
                for player_id, player_stats in stats_dict.items():
                    for k, v in player_stats.items():
                        if k in ['preflop', 'flop', 'turn', 'river']:
                            self.stats[int(player_id)]['bet_sizing'][k] = deque(v, maxlen=100)
                        else:
                            self.stats[int(player_id)][k] = v
            except Exception as e:
                logging.error(f"Ошибка при загрузке статистики: {e}")

# ======================
# Улучшенная обработка состояния
# ======================
class StateProcessor:
    def __init__(self):
        self.kmeans = self._init_kmeans()
        self.state_size = 104 + 64 + 64  # карты + кластер + оппонент-статы
        self.opponent_stats = OpponentStats()

    def _init_kmeans(self):
        if os.path.exists(config.kmeans_path):
            return joblib.load(config.kmeans_path)
        return self._train_kmeans()

    def _train_kmeans(self):
        game = pyspiel.load_game(config.game_name)
        samples = []
        for _ in range(1000):
            state = game.new_initial_state()
            while not state.is_terminal():
                obs = state.information_state_tensor()
                samples.append(obs[:104])
                state.apply_action(np.random.choice(state.legal_actions()))
        kmeans = KMeans(n_clusters=64).fit(samples)
        joblib.dump(kmeans, config.kmeans_path)
        return kmeans

    def process(self, obs_batch):
        processed = []
        for obs in obs_batch:
            cards = obs["info_state"][:104]
            cluster = self.kmeans.predict([cards])[0]
            position = obs["position"] / config.num_players

            # Получаем статистику оппонентов (среднее по всем кроме текущего)
            opponent_features = []
            for i in range(config.num_players):
                if i != obs["current_player"]:
                    opponent_features.append(self.opponent_stats.get_features(i))
            if opponent_features:
                opponent_features = np.mean(opponent_features, axis=0)
            else:
                opponent_features = np.zeros(64)

            features = np.concatenate([
                cards,
                np.eye(64)[cluster],
                opponent_features
            ])
            processed.append(features)
        return np.array(processed, dtype=np.float32)

# ======================
# Улучшенная нейросетевая модель
# ======================
class HybridPokerModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.processor = StateProcessor()

        self.feature_net = nn.Sequential(
            nn.Linear(self.processor.state_size, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Головные слои для разных типов оппонентов
        self.tag_head = nn.Linear(256, action_space.n)
        self.lag_head = nn.Linear(256, action_space.n)
        self.fish_head = nn.Linear(256, action_space.n)
        self.critic = nn.Linear(256, 1)

        self.use_amp = True
        self.autocast_device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, input_dict, state, seq_lens):
        with torch.autocast(self.autocast_device, enabled=self.use_amp):
            processed = self.processor.process(input_dict["obs"])
            x = torch.tensor(processed).float().to(self.device)
            features = self.feature_net(x)

            # Последние 64 признака — статистика оппонентов
            opponent_stats = x[:, -64:]

            # Получаем веса для разных стратегий
            weights = self._get_opponent_weights(opponent_stats)
            tag_out = self.tag_head(features)
            lag_out = self.lag_head(features)
            fish_out = self.fish_head(features)

            output = (
                weights[:, 0:1] * tag_out +
                weights[:, 1:2] * lag_out +
                weights[:, 2:3] * fish_out
            )

            self._value_out = self.critic(features).squeeze(1)
            return output, state

    def _get_opponent_weights(self, stats):
        # Более продвинутая эвристика для определения типа оппонента
        vpip = stats[:, 0]
        pfr = stats[:, 1]
        af = stats[:, 2]

        tag_weight = torch.sigmoid(2.0 - vpip + pfr + af)
        lag_weight = torch.sigmoid(vpip + pfr + af - 1.0)
        fish_weight = torch.sigmoid(vpip - pfr - af)

        weights = torch.stack([tag_weight, lag_weight, fish_weight], dim=1)
        return torch.softmax(weights, dim=1)

    def value_function(self):
        return self._value_out

# ======================
# Окружение с логированием и тестированием
# ======================
class PokerEnv(rl_environment.Environment):
    def __init__(self):
        super().__init__(config.game_name)
        self._position_names = ['SB', 'BB', 'UTG1', 'UTG2', 'UTG3', 'MP', 'CO', 'BU']
        self.stats = OpponentStats()
        self.opponent_type = "Unknown"
        self.episode_rewards = []
        self.episode_actions = []

    def get_position_name(self, player_id):
        return self._position_names[player_id]

    def reset_episode_stats(self):
        self.episode_rewards = []
        self.episode_actions = []

    def log_episode(self, reward, actions):
        self.episode_rewards.append(reward)
        self.episode_actions.append(actions)
        # Можно сохранять в файл или базу

# ======================
# RLlib конфиг и запуск
# ======================
def update_exploit_metrics(info):
    episode = info["episode"]
    env = info["env"]
    reward = episode.total_reward
    episode.custom_metrics["winrate"] = reward
    # Можно добавить свои метрики по VPIP, PFR и т.д.

def setup_training():
    ray.init(
        num_cpus=12,
        num_gpus=1,
        memory=85*1024**3,
        object_store_memory=20*1024**3,
        ignore_reinit_error=True
    )

    train_config = (
        PPOConfig()
        .environment(
            env=PokerEnv,
            env_config={},
            disable_env_checking=True
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=config.num_workers,
            num_envs_per_worker=config.num_envs_per_worker,
            rollout_fragment_length=config.rollout_fragment_length,
            batch_mode="truncate_episodes",
            remote_worker_envs=True
        )
        .training(
            model={"custom_model": HybridPokerModel},
            train_batch_size=config.batch_size,
            sgd_minibatch_size=config.sgd_minibatch_size,
            num_sgd_iter=config.num_sgd_iter,
            gamma=config.gamma,
            lr=config.lr,
            use_amp=True,
            clip_param=0.2,
            lambda_=0.95,
            _enable_learner_api=True
        )
        .resources(
            num_gpus=config.gpu_fraction,
            num_cpus_per_worker=1,
            num_gpus_per_worker=0.05
        )
        .reporting(
            metrics_num_episodes_for_smoothing=1000,
            min_time_s_per_iteration=30
        )
        .callbacks({
            "on_episode_end": update_exploit_metrics,
        })
    )
    return train_config

def train_poker():
    train_config = setup_training()
    tune.run(
        "PPO",
        config=train_config.to_dict(),
        stop={"timesteps_total": config.max_training_steps},
        checkpoint_freq=config.checkpoint_freq,
        local_dir=str(config.base_dir),
        verbose=3,
        reuse_actors=True
    )

# ======================
# Тестирование против фиксированных ботов
# ======================
def evaluate_against_bots(agent_model, env_class, num_episodes=1000):
    results = {"TAG": [], "LAG": [], "Fish": []}
    for bot_type in results.keys():
        for _ in range(num_episodes):
            env = env_class()
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                obs = state["obs"]
                # Получаем действие от модели
                action, _ = agent_model.forward({"obs": [obs]}, None, None)
                action = int(torch.argmax(action).item())
                # Получаем действие от бота
                if bot_type == "TAG":
                    bot_action = TAG(env).action_probabilities(state, player_id=1)
                elif bot_type == "LAG":
                    bot_action = LAG(env).action_probabilities(state, player_id=1)
                else:
                    bot_action = Fish(env).action_probabilities(state, player_id=1)
                # Применяем действия
                state, reward, done, info = env.step([action, bot_action])
                total_reward += reward[0]
            results[bot_type].append(total_reward)
    # Сохраняем результаты
    df = pd.DataFrame(results)
    df.to_csv(config.metrics_path)
    print("Evaluation results saved:", config.metrics_path)

# ======================
# Запуск
# ======================
if __name__ == "__main__":
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    train_poker()
    # После обучения можно вызвать:
    # evaluate_against_bots(agent_model, PokerEnv, num_episodes=1000)

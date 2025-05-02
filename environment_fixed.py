import logging
import os
import random
import time
import traceback
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from open_spiel.python import rl_environment

# Удаляем импорт get_legal_actions_map, так как он не нужен
# эта функция используется только в OpenSpiel для других целей

# Настройка логгера
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class OpponentStats:
    """Класс для хранения и обновления базовой статистики оппонентов."""
    
    def __init__(self, num_players, dtype=np.float32):
        """
        Инициализирует статистику для указанного количества игроков.
        
        Args:
            num_players: Количество игроков в игре
            dtype: Тип данных для статистики
        """
        self.num_players = num_players
        self.dtype = dtype
        self.stats = {}
        self.reset()
    
    def reset(self):
        """Сбрасывает всю статистику."""
        self.stats = {}
        for player_id in range(self.num_players):
            self._ensure_player_stats(player_id)
    
    def _ensure_player_stats(self, player_id):
        """Создает структуру статистики для игрока, если она отсутствует."""
        if player_id not in self.stats:
            self.stats[player_id] = {
                # Базовые счетчики
                "hands_played": 0,
                "vpip_opportunities": 0,  # Возможности для VPIP
                "vpip_actions": 0,        # Действия VPIP (добровольно вложено)
                "pfr_opportunities": 0,    # Возможности для PFR
                "pfr_actions": 0,          # Действия PFR (рейз на префлопе)
                
                # Счетчики по улицам (0=preflop, 1=flop, 2=turn, 3=river)
                "street_opportunities": [0, 0, 0, 0],  # Возможности по улицам
                "street_calls": [0, 0, 0, 0],          # Колы по улицам
                "street_raises": [0, 0, 0, 0],         # Рейзы по улицам
                "street_folds": [0, 0, 0, 0],          # Фолды по улицам
                
                # Флаги для текущей руки
                "current_hand_vpip": False,  # Был ли VPIP в текущей руке
                "current_hand_pfr": False,   # Был ли PFR в текущей руке
                "current_hand_involved": False,  # Вовлечен ли в текущую руку
            }
    
    def get_features(self, player_id):
        """Возвращает нормализованную статистику для игрока как numpy array нужного dtype."""
        self._ensure_player_stats(player_id)
        stats = self.stats[player_id]
        
        # Расчет основных показателей
        hands = max(1, stats["hands_played"])  # Избегаем деления на ноль
        vpip = stats["vpip_actions"] / max(1, stats["vpip_opportunities"])
        pfr = stats["pfr_actions"] / max(1, stats["pfr_opportunities"])
        
        # Расчет показателей по улицам
        street_stats = []
        for i in range(4):  # 4 улицы
            opps = max(1, stats["street_opportunities"][i])
            call_freq = stats["street_calls"][i] / opps
            raise_freq = stats["street_raises"][i] / opps
            fold_freq = stats["street_folds"][i] / opps
            street_stats.extend([call_freq, raise_freq, fold_freq])
        
        # Объединение всех признаков
        features = np.array([
            vpip, pfr,
            *street_stats,
            stats["hands_played"] / 100.0,  # Нормализуем количество рук (до 100)
        ], dtype=self.dtype)
        
        return features
    
    def record_opportunity(self, player_id, is_vpip_opp, is_pfr_opp):
        """Записывает возможность для VPIP/PFR."""
        self._ensure_player_stats(player_id)
        if is_vpip_opp:
            self.stats[player_id]["vpip_opportunities"] += 1
        if is_pfr_opp:
            self.stats[player_id]["pfr_opportunities"] += 1
    
    def update_on_action(self, player_id, action_type, street, is_voluntary_action):
        """
        Обновляет счетчики действий.
        
        Args:
            player_id: ID игрока
            action_type: Тип действия (0=fold, 1=call/check, 2=raise)
            street: Текущая улица (0=preflop, 1=flop, 2=turn, 3=river)
            is_voluntary_action: Является ли действие добровольным (не BB/SB)
        """
        self._ensure_player_stats(player_id)
        stats = self.stats[player_id]
        
        # Обновление счетчиков действий по улицам
        if 0 <= street < 4:  # Проверка валидности улицы
            stats["street_opportunities"][street] += 1
            
            if action_type == 0:  # fold
                stats["street_folds"][street] += 1
            elif action_type == 1:  # call/check
                stats["street_calls"][street] += 1
            elif action_type == 2:  # raise
                stats["street_raises"][street] += 1
        
        # Обновление VPIP и PFR если это преффлоп
        if street == 0:  # preflop
            # Добровольное вложение в банк
            if is_voluntary_action and (action_type == 1 or action_type == 2):
                stats["vpip_actions"] += 1
                stats["current_hand_vpip"] = True
            
            # Рейз на префлопе
            if action_type == 2:
                stats["pfr_actions"] += 1
                stats["current_hand_pfr"] = True
        
        # Отмечаем, что игрок вовлечен в руку
        stats["current_hand_involved"] = True
    
    def finalize_hand_stats(self, involved_players):
        """
        Вызывается в конце руки для увеличения счетчика рук и сброса флагов.
        
        Args:
            involved_players: Список ID игроков, вовлеченных в руку
        """
        for player_id in range(self.num_players):
            self._ensure_player_stats(player_id)
            stats = self.stats[player_id]
            
            # Увеличиваем счетчик сыгранных рук
            if player_id in involved_players:
                stats["hands_played"] += 1
            
            # Сбрасываем флаги для следующей руки
            stats["current_hand_vpip"] = False
            stats["current_hand_pfr"] = False
            stats["current_hand_involved"] = False


class PokerEnv(gym.Env):
    """
    Среда покера для RLlib, использующая OpenSpiel и Action Masking.
    Observation space: Dict("obs": Box, "action_mask": Box)
    """
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}
    
    # Имена действий для удобства отладки
    ACTION_NAMES = {
        0: "fold",
        1: "call/check",
        2: "raise"
    }
    
    def __init__(self, env_config: dict):
        """
        Инициализирует среду покера.
        
        Args:
            env_config: Словарь конфигурации среды
        """
        super().__init__()
        
        # Параметры игры
        self.game_name = env_config.get("game_name", "universal_poker")
        self.num_players = env_config.get("num_players", 2)
        
        # Параметры стеков и блайндов
        self.starting_stack = env_config.get("starting_stack", 10000)
        self.small_blind = env_config.get("small_blind", 50)
        self.big_blind = env_config.get("big_blind", 100)
        
        # Параметры для определения OpenSpiel
        game_config = env_config.get("game_config", {})
        self.game_config = {
            "numPlayers": self.num_players,
            "betting": "limit",  # nolimit or limit
            "numRounds": 4,
            "numSuits": 4,
            "numRanks": 13,
            "numHoleCards": 2,
            "numBoardCards": [0, 3, 1, 1],  # 0 на префлопе, 3 на флопе, 1 на терне, 1 на ривере
            "blind": f"{self.small_blind} {self.big_blind}",
            "raiseSize": "100 100 200 400",  # Для limit
            "maxRaises": "3 4 4 4",  # Максимальное количество рейзов по улицам
            "stack": self.starting_stack,
        }
        
        # Обновление из внешней конфигурации
        self.game_config.update(game_config)
        
        # Тип данных для наблюдений
        self.dtype = env_config.get("dtype", np.float32)
        
        # Создание базовой среды OpenSpiel
        self.game = rl_environment.Environment(self.game_name, **self._process_game_params(self.game_config))
        self._base_env = self.game
        
        # Статистика оппонентов
        self.stats = OpponentStats(self.num_players, dtype=self.dtype)
        
        # Определение размерности базового наблюдения
        first_obs = self.game.reset()
        self._base_obs_size = len(first_obs.observations["info_state"][0])
        
        # Размерность статистики оппонентов (на одного оппонента)
        self._opp_stats_size = 14  # vpip, pfr, 3 метрики * 4 улицы, кол-во рук
        
        # Общая размерность статистики всех оппонентов
        self._stats_size = self._opp_stats_size * (self.num_players - 1)
        
        # Общая размерность наблюдения
        self._total_obs_size = self._base_obs_size + self._stats_size
        
        # Определение пространств наблюдения и действий
        self._define_spaces(self.num_players)
        
        # Для отслеживания текущей руки и улицы
        self._current_time_step = None
        self._episode_ended = False
        self._round_num = 0
        self._last_action = None
        
        # Метрики для оценки качества агента
        self._reset_metrics()
        
        logger.info(f"Initialized PokerEnv with {self.num_players} players, obs_size={self._total_obs_size}")
    
    def _process_game_params(self, game_params):
        """
        Приводит параметры игры к формату, ожидаемому OpenSpiel.
        Некоторые параметры требуют специальной обработки.
        """
        processed = {}
        for key, value in game_params.items():
            if key == "blind" and isinstance(value, (list, tuple)):
                processed[key] = " ".join(map(str, value))
            elif key == "numBoardCards" and isinstance(value, (list, tuple)):
                processed[key] = " ".join(map(str, value))
            else:
                processed[key] = value
        return processed
    
    def seed(self, seed=None):
        """Устанавливает seed для генератора случайных чисел среды."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        return [seed]
    
    def _reset_metrics(self):
        """Сбрасывает метрики для нового эпизода."""
        self._episode_reward = 0
        self._episode_length = 0
        self._vpip_rate = 0
        self._pfr_rate = 0
        self._action_counts = {0: 0, 1: 0, 2: 0}  # fold, call/check, raise
        self._involved_players = set()
    
    def _define_spaces(self, num_players):
        """Определяет action_space и observation_space (Dict) для Gymnasium."""
        # Пространство действий: 0 (fold), 1 (call/check), 2 (raise)
        self.action_space = spaces.Discrete(3)
        
        # Пространство наблюдений: словарь с наблюдением и маской действий
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self._total_obs_size,), 
                dtype=self.dtype
            ),
            "action_mask": spaces.Box(
                low=0, high=1,
                shape=(self.action_space.n,),
                dtype=np.int8
            )
        })
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Сбрасывает среду для нового эпизода.
        
        Args:
            seed: Optional seed для генератора случайных чисел
            options: Дополнительные параметры
            
        Returns:
            Начальное наблюдение и info
        """
        try:
            # Установка seed, если указан
            if seed is not None:
                self.seed(seed)
            
            # Сброс метрик
            self._reset_metrics()
            
            # Сброс базовой среды
            self._current_time_step = self.game.reset()
            self._episode_ended = False
            self._round_num = 0
            self._last_action = None
            
            # Сброс множества вовлеченных игроков
            self._involved_players = set()
            
            # Получение начального наблюдения
            observation = self._get_observation()
            
            # Информация о состоянии среды
            info = self._get_info()
            
            return observation, info
            
        except Exception as e:
            logger.error(f"Error in reset: {e}")
            logger.error(traceback.format_exc())
            # Аварийное восстановление
            dummy_obs = {
                "obs": np.zeros(self._total_obs_size, dtype=self.dtype),
                "action_mask": np.ones(self.action_space.n, dtype=np.int8)
            }
            return dummy_obs, {}
    
    def _current_player(self):
        """Получает ID текущего игрока."""
        if self._current_time_step is None:
            return -1
        return self._current_time_step.observations["current_player"]
    
    def is_terminal(self):
        """Проверяет, является ли текущее состояние терминальным."""
        if self._current_time_step is None:
            return True
        return self._current_time_step.last()
    
    def _get_legal_actions_mask(self):
        """
        Получает маску легальных действий.
        1 = легальное действие, 0 = нелегальное действие
        """
        if self._current_time_step is None or self.is_terminal():
            # В терминальном состоянии нет легальных действий
            return np.zeros(self.action_space.n, dtype=np.int8)
        
        # Получаем карту легальных действий из OpenSpiel
        legal_actions_mask = np.zeros(self.action_space.n, dtype=np.int8)
        curr_player = self._current_player()
        legal_actions = self._current_time_step.observations["legal_actions"][curr_player]
        
        # OpenSpiel использует другие коды действий, приводим к нашим
        for action in legal_actions:
            # Fold = 0, Call = 1, Raise = 2 (в нашей среде)
            if action == 0:  # FOLD
                legal_actions_mask[0] = 1
            elif action == 1:  # CALL
                legal_actions_mask[1] = 1
            elif action == 2:  # RAISE
                legal_actions_mask[2] = 1
        
        return legal_actions_mask
    
    def _get_observation(self):
        """
        Получает словарь наблюдения с 'obs' и 'action_mask'.
        """
        try:
            # Получаем маску легальных действий
            action_mask = self._get_legal_actions_mask()
            
            # Получаем текущего игрока
            player_id = self._current_player()
            if player_id < 0:
                logger.warning(f"Invalid player ID: {player_id}, using fallback observation")
                # Возвращаем заполненное нулями наблюдение
                return {
                    "obs": np.zeros(self._total_obs_size, dtype=self.dtype),
                    "action_mask": action_mask
                }
                
            # Получаем базовое наблюдение (info_state) из OpenSpiel
            if "info_state" not in self._current_time_step.observations:
                logger.warning("info_state not found in observations, using fallback")
                info_state = np.zeros(self._base_obs_size, dtype=self.dtype)
            else:
                info_state = np.array(
                    self._current_time_step.observations["info_state"][player_id],
                    dtype=self.dtype
                )
                
            # Проверка размерности
            if len(info_state) != self._base_obs_size:
                logger.warning(f"Unexpected info_state size: got {len(info_state)}, expected {self._base_obs_size}")
                # Адаптируем размер
                if len(info_state) > self._base_obs_size:
                    info_state = info_state[:self._base_obs_size]
                else:
                    pad_size = self._base_obs_size - len(info_state)
                    info_state = np.pad(info_state, (0, pad_size))
            
            # Собираем статистику оппонентов
            opponents_stats = []
            for opp_id in range(self.num_players):
                if opp_id != player_id:
                    opponents_stats.append(self.stats.get_features(opp_id))
                    
            # Объединяем все оппонентов в одно наблюдение
            opponent_features = np.concatenate(opponents_stats) if opponents_stats else np.zeros(self._stats_size, dtype=self.dtype)
            
            # Объединяем базовое наблюдение и статистику оппонентов
            observation = np.concatenate([info_state, opponent_features])
            
            return {
                "obs": observation.astype(self.dtype),
                "action_mask": action_mask
            }
        except Exception as e:
            logger.error(f"Error getting observation: {e}")
            logger.error(traceback.format_exc())
            # Возвращаем безопасное наблюдение в случае ошибки
            return {
                "obs": np.zeros(self._total_obs_size, dtype=self.dtype),
                "action_mask": np.ones(self.action_space.n, dtype=np.int8)  # Разрешаем все действия
            }
    
    def step(self, action: int):
        """
        Выполняет шаг в среде.
        
        Args:
            action: Действие для выполнения
            
        Returns:
            observation: Новое наблюдение
            reward: Полученная награда
            terminated: Завершен ли эпизод
            truncated: Был ли эпизод прерван (не используется)
            info: Дополнительная информация
        """
        try:
            if self._episode_ended:
                logger.warning("Episode already ended, reset needed")
                observation = self._get_observation()
                return observation, 0.0, True, False, self._get_info()
            
            # Получаем ID текущего игрока и улицу перед действием
            player_id = self._current_player()
            street_before = self._get_street(self._current_time_step)
            
            # Обновляем информацию перед действием
            self._update_hand_info_pre_action(player_id, action)
            
            # Конвертируем наше действие в действие OpenSpiel
            spiel_action = action  # В нашем случае совпадает
            
            # Выполняем действие в среде
            self._last_action = action
            self._current_time_step = self.game.step([spiel_action])
            
            # Обновляем статистику после действия
            self._update_statistics(player_id, action)
            
            # Улица после действия
            street_after = self._get_street(self._current_time_step)
            
            # Если изменилась улица, обновляем _round_num
            if street_after != street_before:
                self._round_num = street_after
            
            # Обновляем метрики эпизода
            self._episode_length += 1
            self._action_counts[action] += 1
            
            # Если это терминальное состояние, финализируем статистику руки
            if self.is_terminal():
                self._episode_ended = True
                self.stats.finalize_hand_stats(self._involved_players)
            
            # Получаем награду, наблюдение и info
            rewards = self._current_time_step.rewards
            # Текущая награда - для текущего игрока
            reward = rewards[player_id] if player_id < len(rewards) else 0.0
            self._episode_reward += reward
            
            observation = self._get_observation()
            info = self._get_info()
            
            return observation, reward, self._episode_ended, False, info
            
        except Exception as e:
            logger.error(f"Error in step: {e}")
            logger.error(traceback.format_exc())
            # Аварийное восстановление
            observation = self._get_observation()
            return observation, 0.0, True, False, self._get_info()
    
    def _update_hand_info_pre_action(self, player_id, action):
        """Обновляет информацию о руке перед выполнением действия."""
        if player_id >= 0:
            # Добавляем игрока в множество вовлеченных в руку
            self._involved_players.add(player_id)
            
            # Получаем улицу (0=preflop, 1=flop и т.д.)
            street = self._get_street(self._current_time_step)
            
            # Определяем, является ли действие добровольным
            # (не защита блайндов или проверка BB)
            is_voluntary = True  # Большинство действий добровольные
            
            # Обновляем статистику для оппонентов
            is_vpip_opp = (street == 0)  # Возможность VPIP только на префлопе
            is_pfr_opp = (street == 0)   # Возможность PFR только на префлопе
            
            if is_vpip_opp or is_pfr_opp:
                self.stats.record_opportunity(player_id, is_vpip_opp, is_pfr_opp)
    
    def _update_statistics(self, player_id, action):
        """Обновляет статистику игрока на основе выполненного действия."""
        if player_id >= 0:
            # Получаем улицу (0=preflop, 1=flop и т.д.)
            street = self._get_street(self._current_time_step)
            
            # Определяем тип действия (в нашей кодировке)
            action_type = action  # fold=0, call/check=1, raise=2
            
            # Определяем, является ли действие добровольным для VPIP
            is_voluntary = True  # Большинство действий добровольные
            
            # Обновляем статистику
            self.stats.update_on_action(player_id, action_type, street, is_voluntary)
    
    def _get_info(self):
        """Получает дополнительную информацию о состоянии среды."""
        player_id = self._current_player()
        street = self._get_street(self._current_time_step)
        position = self._get_position_name(player_id)
        
        return {
            "player_id": player_id,
            "street": street,
            "position": position,
            "pot_size": self._get_pot_size(self._current_time_step),
            "episode_length": self._episode_length,
            "episode_reward": self._episode_reward,
            "action_counts": self._action_counts,
        }
    
    def _get_position_name(self, player_id):
        """Получает название позиции для ID игрока."""
        if player_id < 0 or player_id >= self.num_players:
            return "INVALID"
        
        positions = ["SB", "BB", "UTG", "MP", "CO", "BTN"]
        if self.num_players == 2:
            return ["BTN/SB", "BB"][player_id]
        
        position_idx = (player_id - 2) % self.num_players
        if position_idx < len(positions) - 2:
            return positions[position_idx + 2]
        return positions[position_idx - len(positions) + 2]

    def _get_street(self, time_step):
        """Получает текущую улицу."""
        state = self._get_state()
        return int(state.round()) if state and hasattr(state, 'round') else 0
    
    def _get_state(self):
        """Получает внутреннее состояние OpenSpiel."""
        return getattr(self._base_env, '_state', None)
    
    def _get_pot_size(self, time_step):
        """Получает размер банка."""
        state = self._get_state()
        return float(state.pot()) if state and hasattr(state, 'pot') else 0.0
    
    def render(self, mode='human'):
        """
        Отображает текущее состояние игры.
        
        Args:
            mode: Режим отображения ('human' или 'ansi')
            
        Returns:
            Строка с отображением или None
        """
        if self._current_time_step is None:
            return "Game not started"
        
        player_id = self._current_player()
        state = self._get_state()
        
        if mode == 'ansi':
            return str(state) if state else "No state available"
        
        return None
    
    def close(self):
        """Освобождает ресурсы."""
        pass

def create_poker_env(env_config):
    """Фабричная функция для создания среды покера."""
    return PokerEnv(env_config)
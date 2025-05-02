#!/usr/bin/env python3
"""
Улучшенное окружение для покера на основе OpenSpiel.
Отладочная версия с более надежной обработкой параметров.
"""

import random
import traceback
import logging
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from open_spiel.python import rl_environment
import pyspiel

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("poker_env")

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
                "vpip_count": 0,  # Количество добровольных входов в банк
                "vpip_opp": 0,  # Количество возможностей для VPIP
                "pfr_count": 0,  # Количество рейзов префлоп
                "pfr_opp": 0,  # Количество возможностей для PFR
                "hands_played": 0,  # Количество сыгранных рук
                
                # Действия по улицам (fold, call/check, raise)
                "actions": {
                    0: [0, 0, 0],  # Префлоп
                    1: [0, 0, 0],  # Флоп
                    2: [0, 0, 0],  # Терн
                    3: [0, 0, 0],  # Ривер
                },
                
                # Флаги для текущей руки
                "this_hand": {
                    "vpip": False,  # Вошел ли игрок в банк добровольно
                    "pfr": False,  # Сделал ли игрок рейз префлоп
                }
            }
    
    def get_features(self, player_id):
        """Возвращает нормализованную статистику для игрока как numpy array нужного dtype."""
        self._ensure_player_stats(player_id)
        stats = self.stats[player_id]
        
        features = []
        
        # VPIP (Voluntarily Put $ In Pot)
        vpip = stats["vpip_count"] / max(1, stats["vpip_opp"])
        features.append(vpip)
        
        # PFR (PreFlop Raise)
        pfr = stats["pfr_count"] / max(1, stats["pfr_opp"])
        features.append(pfr)
        
        # Частота действий на разных улицах
        for street in range(4):  # 0=preflop, 1=flop, 2=turn, 3=river
            actions = stats["actions"][street]
            total = sum(actions)
            if total > 0:
                fold_freq = actions[0] / total
                call_freq = actions[1] / total
                raise_freq = actions[2] / total
            else:
                fold_freq = call_freq = raise_freq = 0
            
            features.extend([fold_freq, call_freq, raise_freq])
        
        # Количество сыгранных рук (нормализовано до 1)
        features.append(min(1.0, stats["hands_played"] / 100))
        
        return np.array(features, dtype=self.dtype)
    
    def record_opportunity(self, player_id, is_vpip_opp, is_pfr_opp):
        """Записывает возможность для VPIP/PFR."""
        self._ensure_player_stats(player_id)
        
        if is_vpip_opp:
            self.stats[player_id]["vpip_opp"] += 1
        
        if is_pfr_opp:
            self.stats[player_id]["pfr_opp"] += 1
    
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
        
        # Увеличиваем счетчик действия
        self.stats[player_id]["actions"][street][action_type] += 1
        
        # Обновляем VPIP и PFR, если это добровольное действие
        if is_voluntary_action:
            # Call или Raise считаются как вход в банк
            if action_type in [1, 2]:
                if not self.stats[player_id]["this_hand"]["vpip"]:
                    self.stats[player_id]["this_hand"]["vpip"] = True
                    self.stats[player_id]["vpip_count"] += 1
            
            # Только рейз на префлопе учитывается для PFR
            if street == 0 and action_type == 2:
                if not self.stats[player_id]["this_hand"]["pfr"]:
                    self.stats[player_id]["this_hand"]["pfr"] = True
                    self.stats[player_id]["pfr_count"] += 1
    
    def finalize_hand_stats(self, involved_players):
        """
        Вызывается в конце руки для увеличения счетчика рук и сброса флагов.
        
        Args:
            involved_players: Список ID игроков, вовлеченных в руку
        """
        for player_id in range(self.num_players):
            self._ensure_player_stats(player_id)
            
            # Увеличиваем счетчик рук
            self.stats[player_id]["hands_played"] += 1
            
            # Сбрасываем флаги для следующей руки
            self.stats[player_id]["this_hand"] = {
                "vpip": False,
                "pfr": False,
            }


class PokerEnv(gym.Env):
    """
    Среда покера для RLlib, использующая OpenSpiel и Action Masking.
    Observation space: Dict("obs": Box, "action_mask": Box)
    """
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}
    
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
        
        # Логирование инициализации
        logger.info(f"Initializing PokerEnv with config: {env_config}")
        
        # Извлекаем конфигурацию из переданного словаря
        self.game_name = env_config.get("game_name", "universal_poker")
        self.num_players = env_config.get("num_players", 2)
        self.starting_stack = env_config.get("starting_stack", 10000)
        self.small_blind = env_config.get("small_blind", 50)
        self.big_blind = env_config.get("big_blind", 100)
        
        # Дополнительная конфигурация покерной игры
        game_config = env_config.get("game_config", {})
        
        # Создаем базовую конфигурацию с правильными параметрами
        # Внимание: numBoardCards и stack требуют особого внимания
        self.game_config = {
            "numPlayers": self.num_players,
            "betting": "nolimit",
            "numRounds": 4,
            "numSuits": 4,
            "numRanks": 13,
            "numHoleCards": 2,
            "numBoardCards": "0 3 1 1",  # 0 на префлопе, 3 на флопе, 1 на терне, 1 на ривере
            "blind": f"{self.small_blind} {self.big_blind}" + " 0" * (self.num_players - 2),  # SB, BB, остальные 0
            "raiseSize": "100 100 200 400",  # Для limit
            "maxRaises": "3 4 4 4",  # Максимальное количество рейзов по улицам
        }
        
        # Добавляем стеки игроков
        stack_values = [self.starting_stack] * self.num_players
        self.game_config["stack"] = " ".join(map(str, stack_values))
        
        # Обновление из внешней конфигурации
        if game_config:
            logger.info(f"Updating game config with: {game_config}")
            self.game_config.update(game_config)
        
        # Проверяем и корректируем формат numBoardCards, если нужно
        if isinstance(self.game_config.get("numBoardCards"), list):
            self.game_config["numBoardCards"] = " ".join(map(str, self.game_config["numBoardCards"]))
        
        # Тип данных для наблюдений
        self.dtype = env_config.get("dtype", np.float32)
        
        # Преобразуем все параметры в строковый формат, как ожидает OpenSpiel
        processed_config = self._process_game_params(self.game_config)
        logger.info(f"Processed game config: {processed_config}")
        
        try:
            # Создание базовой среды OpenSpiel
            self.game = rl_environment.Environment(self.game_name, **processed_config)
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
            
            logger.info(f"Successfully initialized PokerEnv with {self.num_players} players, obs_size={self._total_obs_size}")
            
        except Exception as e:
            logger.error(f"Error initializing PokerEnv: {e}")
            logger.error(traceback.format_exc())
            # Создаем минимальную конфигурацию для аварийного восстановления
            # В этом случае среда может не работать правильно, но не вызовет краш
            self._base_obs_size = 1000  # Предполагаемый размер
            self._opp_stats_size = 14
            self._stats_size = self._opp_stats_size * (self.num_players - 1)
            self._total_obs_size = self._base_obs_size + self._stats_size
            self._define_spaces(self.num_players)
            self._reset_metrics()
            self._current_time_step = None
            self._episode_ended = True
    
    def _process_game_params(self, game_params):
        """
        Приводит параметры игры к формату, ожидаемому OpenSpiel.
        Некоторые параметры требуют специальной обработки.
        """
        processed = {}
        for key, value in game_params.items():
            if isinstance(value, (list, tuple)):
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
            if player_id < 0 or player_id >= self.num_players:
                logger.warning(f"Invalid player ID: {player_id}, using fallback observation")
                # Возвращаем заполненное нулями наблюдение
                return {
                    "obs": np.zeros(self._total_obs_size, dtype=self.dtype),
                    "action_mask": action_mask
                }
                
            # Получаем базовое наблюдение (info_state) из OpenSpiel
            if (self._current_time_step is None or 
                "info_state" not in self._current_time_step.observations or
                player_id >= len(self._current_time_step.observations["info_state"])):
                
                logger.warning(f"info_state not found or invalid index {player_id}, using fallback")
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
            
            # Получаем текущего игрока
            player_id = self._current_player()
            if player_id < 0:
                logger.error(f"Invalid player_id: {player_id}")
                observation = self._get_observation()
                return observation, 0.0, True, False, self._get_info()
            
            # Проверяем легальность действия
            legal_actions = self._current_time_step.observations["legal_actions"][player_id]
            if action not in range(3) or action >= len(legal_actions):
                logger.warning(f"Invalid action {action}, legal actions: {legal_actions}")
                action = legal_actions[0] if legal_actions else 0
            
            # Маппинг действия в OpenSpiel формат
            spiel_action = action
            
            # Обновляем информацию о руке
            self._update_hand_info_pre_action(player_id, action)
            
            # Выполняем действие и получаем новое состояние
            self._current_time_step = self.game.step([spiel_action])
            self._last_action = action
            
            # Обновляем статистику оппонентов
            self._update_statistics(player_id, action)
            
            # Обновляем счетчики
            self._episode_length += 1
            self._action_counts[action] += 1
            
            # Добавляем игрока в множество вовлеченных игроков
            self._involved_players.add(player_id)
            
            # Получаем новое наблюдение
            observation = self._get_observation()
            
            # Определяем, закончился ли эпизод
            terminated = self.is_terminal()
            
            # Определяем награду
            if terminated:
                # Награда в конце эпизода - это выигрыш или проигрыш
                rewards = self._current_time_step.rewards
                reward = rewards[player_id] if player_id < len(rewards) else 0.0
                
                # Завершаем статистику руки
                self.stats.finalize_hand_stats(list(self._involved_players))
            else:
                reward = 0.0  # Промежуточная награда, если рука еще не завершена
            
            # Обновляем общую награду за эпизод
            self._episode_reward += reward
            
            # Получаем дополнительную информацию
            info = self._get_info()
            
            return observation, reward, terminated, False, info
            
        except Exception as e:
            logger.error(f"Error in step: {e}")
            logger.error(traceback.format_exc())
            # Аварийное завершение эпизода
            self._episode_ended = True
            observation = self._get_observation()
            return observation, 0.0, True, False, self._get_info()
    
    def _update_hand_info_pre_action(self, player_id, action):
        """Обновляет информацию о руке перед выполнением действия."""
        try:
            # Получаем текущую улицу (для VPIP и PFR)
            street = self._get_street(self._current_time_step)
            
            # Определяем, является ли действие добровольным
            is_voluntary = True  # По умолчанию считаем действие добровольным
            position = self._get_position_name(player_id)
            
            # Оппорьюнити для VPIP и PFR
            if street == 0 and self._episode_length < 4:  # Префлоп, первый круг торгов
                is_vpip_opp = position not in ["BB"]  # SB обычно учитывается
                is_pfr_opp = True  # Все позиции имеют возможность для PFR
                self.stats.record_opportunity(player_id, is_vpip_opp, is_pfr_opp)
            
        except Exception as e:
            logger.error(f"Error updating hand info: {e}")
    
    def _update_statistics(self, player_id, action):
        """Обновляет статистику игрока на основе выполненного действия."""
        try:
            # Получаем текущую улицу
            street = self._get_street(self._current_time_step)
            
            # Определяем, является ли действие добровольным
            is_voluntary = True  # По умолчанию считаем действие добровольным
            position = self._get_position_name(player_id)
            
            # На префлопе BB и SB имеют особый статус
            if street == 0 and position in ["BB", "SB"] and self._episode_length < 3:
                is_voluntary = False
            
            # Обновляем статистику
            self.stats.update_on_action(player_id, action, street, is_voluntary)
            
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")
    
    def _get_info(self):
        """Получает дополнительную информацию о состоянии среды."""
        info = {
            "episode_length": self._episode_length,
            "episode_reward": self._episode_reward,
            "action_counts": self._action_counts,
        }
        
        # Добавляем информацию о текущем игроке, если эпизод не завершен
        if not self._episode_ended:
            player_id = self._current_player()
            info["current_player"] = player_id
            info["position"] = self._get_position_name(player_id)
            
            # Добавляем информацию о текущей улице и размере банка
            if self._current_time_step:
                info["street"] = self._get_street(self._current_time_step)
                info["pot_size"] = self._get_pot_size(self._current_time_step)
                
                # Добавляем последнее действие, если оно было
                if self._last_action is not None:
                    info["last_action"] = self.ACTION_NAMES[self._last_action]
        
        return info
    
    def _get_position_name(self, player_id):
        """Получает название позиции для ID игрока."""
        position_names = {
            0: "SB",  # Small Blind
            1: "BB",  # Big Blind
        }
        
        if player_id < 2:
            return position_names[player_id]
        
        # Для остальных позиций
        positions = ["UTG", "MP", "CO", "BTN"]
        idx = min(player_id - 2, len(positions) - 1)
        return positions[idx]
    
    def _get_street(self, time_step):
        """Получает текущую улицу."""
        if time_step is None:
            return 0
        
        try:
            # Получаем состояние игры
            state = self._get_state()
            if state:
                # В OpenSpiel улицы нумеруются от 0 (префлоп) до 3 (ривер)
                round_num = state.round()
                return round_num
        except:
            pass
        
        # Если не удалось получить из состояния, используем эвристику
        if self._episode_length < 8:
            return 0  # Префлоп
        elif self._episode_length < 16:
            return 1  # Флоп
        elif self._episode_length < 24:
            return 2  # Терн
        else:
            return 3  # Ривер
    
    def _get_state(self):
        """Получает внутреннее состояние OpenSpiel."""
        try:
            if hasattr(self.game, "_state") and self.game._state:
                return self.game._state
            return None
        except:
            return None
    
    def _get_pot_size(self, time_step):
        """Получает размер банка."""
        # В OpenSpiel сложно получить размер банка напрямую
        # Можно использовать эвристику или состояние игры
        return 0  # Заглушка, требуется реализация
    
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
        
        state = self._get_state()
        if state:
            state_str = str(state)
            if mode == 'ansi':
                return state_str
            else:
                print(state_str)
                return None
        
        return "State information not available"
    
    def close(self):
        """Освобождает ресурсы."""
        pass


def create_poker_env(env_config):
    """Фабричная функция для создания среды покера."""
    return PokerEnv(env_config)
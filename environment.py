# environment.py
import numpy as np
import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium.spaces import Box, Discrete, Dict as GymDict # Используем GymDict для ясности
from open_spiel.python import rl_environment
import pyspiel
from collections import deque
import logging
from typing import Dict, Any, List, Tuple
import traceback

# Настройка логирования
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # Раскомментировать для детальной отладки
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class OpponentStats:
    """Класс для хранения и обновления базовой статистики оппонентов."""
    def __init__(self, num_players, dtype=np.float32):
        self.num_players = num_players
        self.stats = {}
        self.feature_size = 4 # VPIP, PFR, AF, Hands Played
        self.dtype = dtype
        self.reset()

    def reset(self):
        self.stats = {}
        for i in range(self.num_players):
             self._ensure_player_stats(i)

    def _ensure_player_stats(self, player_id):
         player_id_int = int(player_id)
         if player_id_int not in self.stats:
             self.stats[player_id_int] = {
                 'vpip': 0.0, 'pfr': 0.0, 'af': 1.0, 'hands': 0,
                 'agg_actions': 0, 'pass_actions': 0,
                 'vpip_opportunities': 0, 'pfr_opportunities': 0,
                 'vpip_actions': 0, 'pfr_actions': 0,
                 'vpip_acted_this_hand': False, 'pfr_acted_this_hand': False
             }

    def get_features(self, player_id):
        """Возвращает нормализованную статистику для игрока как numpy array нужного dtype."""
        player_id_int = int(player_id)
        self._ensure_player_stats(player_id_int)
        player_stats = self.stats[player_id_int]

        hands = player_stats.get('hands', 0)
        vpip_opps = player_stats.get('vpip_opportunities', 0)
        pfr_opps = player_stats.get('pfr_opportunities', 0)
        vpip_actions = player_stats.get('vpip_actions', 0)
        pfr_actions = player_stats.get('pfr_actions', 0)
        agg_actions = player_stats.get('agg_actions', 0)
        pass_actions = player_stats.get('pass_actions', 0)

        vpip = float(vpip_actions / vpip_opps) if vpip_opps > 0 else 0.0
        pfr = float(pfr_actions / pfr_opps) if pfr_opps > 0 else 0.0
        # Расчет AF с учетом деления на ноль
        af = float(agg_actions / pass_actions) if pass_actions > 0 else (5.0 if agg_actions > 0 else 1.0)

        norm_vpip = np.clip(vpip, 0.0, 1.0)
        norm_pfr = np.clip(pfr, 0.0, 1.0)
        norm_af = np.clip(af / 5.0, 0.0, 1.0) # Масштабируем AF к ~[0,1]
        norm_hands = np.clip(hands / 100.0, 0.0, 1.0) # Нормализуем количество рук (до 100)

        return np.array([norm_vpip, norm_pfr, norm_af, norm_hands], dtype=self.dtype)

    def record_opportunity(self, player_id, is_vpip_opp, is_pfr_opp):
        """Записывает возможность для VPIP/PFR."""
        player_id_int = int(player_id)
        self._ensure_player_stats(player_id_int)
        if is_vpip_opp: self.stats[player_id_int]['vpip_opportunities'] += 1
        if is_pfr_opp: self.stats[player_id_int]['pfr_opportunities'] += 1

    def update_on_action(self, player_id, action_type, street, is_voluntary_action):
        """Обновляет счетчики действий."""
        player_id_int = int(player_id)
        self._ensure_player_stats(player_id_int)
        stats = self.stats[player_id_int]
        # VPIP: Засчитывается первое добровольное действие (не фолд) на префлопе
        if street == 0 and is_voluntary_action and not stats['vpip_acted_this_hand']:
             stats['vpip_actions'] += 1; stats['vpip_acted_this_hand'] = True
        # PFR: Засчитывается первый рейз на префлопе
        if street == 0 and action_type == 'raise' and not stats['pfr_acted_this_hand']:
             stats['pfr_actions'] += 1; stats['pfr_acted_this_hand'] = True
        # AF: Считаем агрессивные (бет/рейз) и пассивные (колл) действия ПОСТФЛОП
        if street > 0:
            if action_type == 'raise': stats['agg_actions'] += 1
            elif action_type == 'call': stats['pass_actions'] += 1 # Только колл, не чек

    def finalize_hand_stats(self, involved_players):
         """Вызывается в конце руки для увеличения счетчика рук и сброса флагов."""
         for player_id in involved_players:
             player_id_int = int(player_id)
             self._ensure_player_stats(player_id_int)
             self.stats[player_id_int]['hands'] += 1
             # Сбрасываем флаги действий в текущей руке для следующей раздачи
             self.stats[player_id_int]['vpip_acted_this_hand'] = False
             self.stats[player_id_int]['pfr_acted_this_hand'] = False


class PokerEnv(gym.Env):
    """
    Среда покера для RLlib, использующая OpenSpiel и Action Masking.
    Observation space: Dict("obs": Box, "action_mask": Box)
    """
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}

    def __init__(self, env_config: dict):
        super().__init__()
        logger.info("Initializing PokerEnv with Action Masking...")

        # Извлечение конфига
        if "config" not in env_config:
             if "env_config" in env_config: env_config = env_config["env_config"]
             else: raise ValueError("Missing 'config' key (PokerConfig instance) in env_config dictionary.")
        if "config" not in env_config: raise ValueError("Missing 'config' key (PokerConfig instance) after checking nested 'env_config'.")

        self.config = env_config["config"]
        # Получаем dtype или используем float32 по умолчанию
        self.dtype = env_config.get("dtype", np.float32)
        if not isinstance(self.dtype, type) or not np.issubdtype(self.dtype, np.floating):
             logger.warning(f"Invalid dtype specified: {self.dtype}. Defaulting to np.float32.")
             self.dtype = np.float32

        # --- OpenSpiel Initialization ---
        game_params_from_config = self.config.game_config if hasattr(self.config, 'game_config') else {}
        game_name = self.config.game_name if hasattr(self.config, 'game_name') else "leduc_poker" # Значение по умолчанию
        processed_game_params = {}
        # Преобразование параметров для OpenSpiel
        for k, v in game_params_from_config.items():
            if k in ["numBoardCards", "blind", "stack"] and isinstance(v, (list, tuple, np.ndarray)):
                processed_game_params[k] = " ".join(map(str, v))
            elif isinstance(v, bool): processed_game_params[k] = str(v).lower()
            elif isinstance(v, (int, float, np.number)): processed_game_params[k] = v
            else: processed_game_params[k] = str(v)

        logger.info(f"Attempting to load OpenSpiel game: '{game_name}' with processed params: {processed_game_params}")
        try:
            self.game = pyspiel.load_game(game_name, processed_game_params)
            logger.info(f"Game '{game_name}' loaded successfully (Num players: {self.game.num_players()}).")
        except Exception as e:
            logger.error(f"Failed to load game '{game_name}': {e}", exc_info=True); raise

        try:
            # Создание базовой среды OpenSpiel RL
            self._base_env = rl_environment.Environment(game=self.game, include_full_state=False)
            logger.info("OpenSpiel RL Environment created successfully.")
            self._action_spec = self._base_env.action_spec()
            self._observation_spec = self._base_env.observation_spec() # Это спецификация info_state
        except Exception as e:
            logger.error(f"Failed to create OpenSpiel RL Environment: {e}", exc_info=True); raise

        num_players_actual = self.game.num_players()
        # Инициализация статистики оппонентов
        self.stats = OpponentStats(num_players_actual, dtype=self.dtype)

        # Определение пространств Gymnasium (action_space, observation_space)
        self._define_spaces(num_players_actual)

        # Вспомогательные атрибуты
        self._position_names = ['SB', 'BB', 'UTG', 'MP1', 'MP2', 'MP3', 'CO', 'BTN'][:num_players_actual]
        self.current_street = 0
        self._current_time_step = None # Хранит текущий TimeStep от OpenSpiel
        # Инициализация генератора случайных чисел Gymnasium
        self.np_random = None
        self.seed() # Инициализируем np_random

        # Добавляем атрибут spec для совместимости с RLlib/Gymnasium>=0.26
        try:
            from gymnasium.envs.registration import EnvSpec
            # Устанавливаем максимальную длину эпизода (подберите значение; 200 обычно достаточно для руки в покере)
            self.spec = EnvSpec(id="PokerEnv-v0", max_episode_steps=200)
            logger.info(f"Set EnvSpec with max_episode_steps={self.spec.max_episode_steps}")
        except ImportError:
            logger.warning("Could not import EnvSpec from gymnasium.envs.registration. max_episode_steps defaulting to infinity.")
            # Создаем простой объект, чтобы избежать AttributeError, если старая версия gym
            self.spec = type('obj', (object,), {'max_episode_steps': float('inf')})()

        logger.info(f"PokerEnv initialized for {num_players_actual} players.")
        logger.info(f"Observation Space: {self.observation_space}")
        logger.info(f"Action Space: {self.action_space}")

    def seed(self, seed=None):
        """Устанавливает seed для генератора случайных чисел среды."""
        self.np_random, seed = seeding.np_random(seed)
        # Note: OpenSpiel environment uses its own internal seeding based on the game object.
        return [seed]

    def _reset_metrics(self):
        """Сбрасывает метрики для нового эпизода."""
        self.episode_rewards = {p: 0.0 for p in range(self.game.num_players())}
        self.current_hand_info = {
            'actions_by_player': {p: [] for p in range(self.game.num_players())},
            'involved_players': set(),
            'pot_size': 0,
            'current_player': -1,
            'last_raiser': -1,
            'can_check': {}
        }
        # self.stats.reset() # Раскомментировать, если статистика нужна только в пределах одной руки

    def _define_spaces(self, num_players):
        """Определяет action_space и observation_space (Dict) для Gymnasium."""
        # 1. Пространство действий
        num_actions = self._action_spec["num_actions"]
        self.action_space = Discrete(num_actions)
        logger.info(f"Action space defined: {self.action_space}")

        # 2. Пространство наблюдений (Dict)
        try:
            # Определяем размер вектора фич ("obs")
            # Получаем ПРИМЕР одного наблюдения от базовой среды
            temp_ts = self._base_env.reset()
            player_id = temp_ts.observations["current_player"]
            while player_id == pyspiel.PlayerId.CHANCE:
                 temp_ts = self._base_env.step([])
                 player_id = temp_ts.observations["current_player"]

            if player_id < 0 or player_id == pyspiel.PlayerId.TERMINAL:
                raise ValueError("Could not get initial player state to determine observation size.")

            # Размер базового вектора (info_state)
            base_obs_size = len(temp_ts.observations["info_state"][player_id])
            if base_obs_size <= 0:
                 raise ValueError(f"Invalid base_obs_size obtained: {base_obs_size}")

            # Размер дополнительных фич
            num_pos_features = 1
            num_pot_odds_features = 1
            num_spr_features = 1
            num_opponent_stats_features = (num_players - 1) * self.stats.feature_size
            num_extra_features = num_pos_features + num_pot_odds_features + num_spr_features + num_opponent_stats_features

            # Общий размер вектора фич
            total_feature_size = base_obs_size + num_extra_features
            logger.info(f"Calculated feature vector size: {total_feature_size} (Base: {base_obs_size}, Extra: {num_extra_features})")

            # Определяем пространство Box для фич
            feature_box = Box(low=-np.inf, high=np.inf, shape=(total_feature_size,), dtype=self.dtype)
            # Определяем пространство Box для маски действий
            mask_box = Box(low=0, high=1, shape=(num_actions,), dtype=np.int8) # Используем int8 для экономии памяти

            # Определяем финальное пространство наблюдений как словарь
            self.observation_space = GymDict({
                "obs": feature_box,
                "action_mask": mask_box
            })
            logger.info(f"Observation space defined: {self.observation_space}")

        except Exception as e:
             logger.error(f"CRITICAL: Failed to determine observation space size: {e}. Cannot proceed.", exc_info=True)
             raise RuntimeError(f"Failed to initialize observation space: {e}")


    def _get_one_observation(self, time_step, player_id) -> Dict[str, np.ndarray]:
         """Собирает СЛОВАРЬ наблюдения {"obs": ..., "action_mask": ...} для игрока."""
         if not hasattr(self, 'observation_space') or self.observation_space is None:
             # Эта ошибка не должна возникать при правильной инициализации
             logger.error("PANIC: _get_one_observation called before observation_space was defined!")
             raise RuntimeError("Observation space is not initialized.")

         feature_shape = self.observation_space["obs"].shape
         mask_shape = self.observation_space["action_mask"].shape
         num_actions = mask_shape[0]

         # Словарь с нулевыми значениями по умолчанию
         zero_obs_dict = {
             "obs": np.zeros(feature_shape, dtype=self.dtype),
             "action_mask": np.zeros(mask_shape, dtype=np.int8)
         }

         # Обработка невалидного player_id
         if player_id < 0 or player_id >= self.game.num_players():
             logger.warning(f"Invalid player_id ({player_id}) requested. Returning zeros obs.")
             # Попытка установить правильную маску, если возможно
             if time_step and "legal_actions" in time_step.observations:
                  try:
                      legal_actions = time_step.observations["legal_actions"].get(player_id, []) # Безопасное получение
                      if legal_actions: zero_obs_dict["action_mask"][legal_actions] = 1
                  except Exception: pass # Игнорируем ошибки при получении маски для невалидного ID
             return zero_obs_dict

         # Сборка наблюдения для валидного игрока
         try:
            # 1. Собираем вектор фич (obs)
            base_obs_list = time_step.observations["info_state"][player_id]
            base_obs = np.array(base_obs_list, dtype=self.dtype)

            position_feature = np.array([float(player_id) / max(1, self.game.num_players() - 1)], dtype=self.dtype)

            opponent_features_list = []
            for pid in range(self.game.num_players()):
                if pid != player_id: opponent_features_list.extend(self.stats.get_features(pid))
            opponent_features = np.array(opponent_features_list, dtype=self.dtype)
            expected_opp_len = (self.game.num_players() - 1) * self.stats.feature_size
            if len(opponent_features) != expected_opp_len: # Проверка и исправление размера (на всякий случай)
                 logger.warning(f"Opponent features length mismatch! Got {len(opponent_features)}, expected {expected_opp_len}. Resizing.")
                 opponent_features = np.resize(opponent_features, expected_opp_len).astype(self.dtype)

            pot_odds_feature = np.array([self._calculate_pot_odds(time_step, player_id)], dtype=self.dtype)
            stack_to_pot_feature = np.array([self._calculate_stack_to_pot(time_step, player_id)], dtype=self.dtype)

            feature_vector = np.concatenate([
                base_obs, position_feature, pot_odds_feature,
                stack_to_pot_feature, opponent_features
            ]).astype(self.dtype) # Финальное приведение типа

            # Проверка финальной длины вектора фич
            if len(feature_vector) != feature_shape[0]:
                 logger.error(f"Feature vector length mismatch after concatenation! Expected {feature_shape[0]}, got {len(feature_vector)}. Returning zeros.")
                 return zero_obs_dict

            # 2. Создаем маску действий (action_mask)
            legal_actions = time_step.observations["legal_actions"][player_id]
            action_mask = np.zeros(num_actions, dtype=np.int8)
            action_mask[legal_actions] = 1 # Устанавливаем 1 для легальных действий

            # 3. Возвращаем словарь
            observation = {
                "obs": feature_vector,
                "action_mask": action_mask
            }
            return observation

         except Exception as e:
            logger.error(f"Error constructing observation dict for player {player_id}: {e}\n{traceback.format_exc()}")
            return zero_obs_dict # Возвращаем нулевой словарь в случае ошибки


    def reset(self, seed=None, options=None):
        """Сбрасывает среду и возвращает начальное наблюдение (словарь) и info."""
        super().reset(seed=seed) # Управляет self.np_random
        logger.debug("Resetting environment...")

        # Запасной вариант нулевого наблюдения
        zero_obs_dict = {
            "obs": np.zeros(self.observation_space["obs"].shape, dtype=self.dtype),
            "action_mask": np.zeros(self.observation_space["action_mask"].shape, dtype=np.int8)
        }
        info = {}

        try:
            self._current_time_step = self._base_env.reset()
            self.current_street = 0
            self._reset_metrics()

            # Запись возможностей VPIP/PFR
            is_vpip_opp, is_pfr_opp = True, True # Упрощенная логика
            for p in range(self.game.num_players()):
                self.stats.record_opportunity(p, is_vpip_opp, is_pfr_opp)

            # Обработка начальных Chance узлов
            while self._current_time_step.observations["current_player"] == pyspiel.PlayerId.CHANCE:
                 logger.debug("Handling initial chance node.")
                 self._current_time_step = self._base_env.step([])

            current_player = self._current_time_step.observations["current_player"]

            if self._current_time_step.last() or current_player < 0:
                logger.warning("Environment terminated immediately after reset. Returning zero obs.")
                # Попытка установить маску, если возможно
                if self._current_time_step and not self._current_time_step.last() and current_player >= 0:
                     try:
                         legal_actions = self._current_time_step.observations["legal_actions"][current_player]
                         zero_obs_dict["action_mask"][legal_actions] = 1
                     except Exception: pass
                obs = zero_obs_dict
            else:
                # Получаем первое наблюдение для активного игрока
                obs = self._get_one_observation(self._current_time_step, current_player)
                info = self._enhance_info({}, current_player) # Добавляем доп. информацию

            logger.debug(f"Reset complete. First player: {current_player}")
            return obs, info # Возвращаем словарь obs

        except Exception as e:
            logger.error(f"Error during environment reset: {e}\n{traceback.format_exc()}")
            info["error"] = f"Reset failed: {e}"
            return zero_obs_dict, info # Возвращаем нулевой словарь


    def step(self, action):
        """Выполняет шаг в среде."""
        action = int(action.item()) if hasattr(action, 'item') else int(action)

        # Запасной вариант нулевого наблюдения
        zero_obs_dict = {
            "obs": np.zeros(self.observation_space["obs"].shape, dtype=self.dtype),
            "action_mask": np.zeros(self.observation_space["action_mask"].shape, dtype=np.int8)
        }
        info = {}

        # Проверка состояния среды
        if self._current_time_step is None or self._current_time_step.last():
            logger.warning("Step called on a terminal or invalid state.")
            last_player = -1
            if self._current_time_step and hasattr(self._current_time_step, 'observations'):
                 last_player = self._current_time_step.observations.get("current_player", -1)
            if last_player >= 0: # Попробуем получить маску для последнего игрока
                 try:
                      legal = self._current_time_step.observations["legal_actions"][last_player]
                      zero_obs_dict["action_mask"][legal] = 1
                 except Exception: pass
            info["error"] = "Step called on terminal state"
            return zero_obs_dict, 0.0, True, False, info # Возвращаем Done=True


        current_player = self._current_time_step.observations["current_player"]
        if current_player < 0 :
             logger.error(f"Invalid current player ID ({current_player}) in step. Returning error state.")
             info["error"] = f"Invalid player ID {current_player}"
             return zero_obs_dict, 0.0, True, False, info # Завершаем эпизод

        logger.debug(f"Player {current_player} attempts action: {action}")

        legal_actions = self._current_time_step.observations["legal_actions"][current_player]

        # --- ПРОВЕРКА ЛЕГАЛЬНОСТИ ДЕЙСТВИЯ ---
        # С включенным action masking модель НЕ ДОЛЖНА выбирать нелегальное действие.
        if action not in legal_actions:
            action_mask_for_log = np.zeros(self.action_space.n, dtype=np.int8)
            action_mask_for_log[legal_actions] = 1
            error_msg = f"ILLEGAL ACTION CHOSEN BY POLICY! Player {current_player} chose {action}, but legal actions are {legal_actions} (mask: {action_mask_for_log}). This should not happen with action masking enabled."
            logger.error(error_msg)
            # Завершаем эпизод с большим штрафом
            obs = self._get_one_observation(self._current_time_step, current_player)
            info = self._enhance_info({'error': error_msg}, current_player)
            return obs, -1000.0, True, False, info # Штраф и Done=True

        # Выполнение легального действия
        try:
            # Обновление статистики и информации о руке
            action_type = self._get_action_type(action)
            is_voluntary = (action_type != 'fold')
            self.stats.update_on_action(current_player, action_type, self.current_street, is_voluntary)
            self.current_hand_info['actions_by_player'][current_player].append(action)
            if action_type != 'fold': self.current_hand_info['involved_players'].add(current_player)
            if action_type == 'raise': self.current_hand_info['last_raiser'] = current_player

            # Шаг в базовой среде OpenSpiel
            self._current_time_step = self._base_env.step([action])

            # Обновление информации о руке после шага
            self.current_hand_info['pot_size'] = self._get_pot_size(self._current_time_step)
            next_player = self._current_time_step.observations["current_player"]
            self.current_hand_info['current_player'] = next_player

            # Обработка Chance узлов
            while self._current_time_step.observations["current_player"] == pyspiel.PlayerId.CHANCE:
                 logger.debug("Handling chance node.")
                 self._current_time_step = self._base_env.step([])
                 next_player = self._current_time_step.observations["current_player"]

            # Обновление улицы (раунда)
            new_street = self._get_street(self._current_time_step)
            if new_street != self.current_street:
                logger.debug(f"Street changed from {self.current_street} to {new_street}")
                self.current_street = new_street

            # Получение результатов шага
            # Награда дается игроку, который ТОЛЬКО ЧТО походил (current_player)
            reward = float(self._current_time_step.rewards[current_player]) if self._current_time_step.rewards else 0.0
            terminated = self._current_time_step.last()
            truncated = False # В покере нет усечения по времени

            # Получение наблюдения для СЛЕДУЮЩЕГО игрока
            obs_player_id = next_player if not terminated else current_player
            if obs_player_id < 0: obs_player_id = current_player # Если next_player == TERMINAL
            obs = self._get_one_observation(self._current_time_step, obs_player_id) # Получаем словарь obs

            # Обновление суммарной награды за эпизод для текущего игрока
            self.episode_rewards[current_player] += reward

            # Формирование дополнительной информации
            info = self._enhance_info({}, obs_player_id) # Info для следующего игрока

            # Завершающие действия, если эпизод закончился
            if terminated:
                logger.debug(f"Episode terminated. Final rewards: {self._current_time_step.rewards}")
                involved_players = self.current_hand_info.get('involved_players', set(range(self.game.num_players())))
                self.stats.finalize_hand_stats(involved_players)
                # Финальные награды заменяют накопленные награды
                final_rewards = {p: float(r) for p, r in enumerate(self._current_time_step.rewards)}
                self.episode_rewards = final_rewards
                info['episode_rewards'] = final_rewards # Добавляем финальные награды в info
                logger.info(f"Episode finished. Total rewards: {self.episode_rewards}")

            return obs, reward, terminated, truncated, info # Возвращаем словарь obs

        except Exception as e:
            logger.error(f"Error during environment step: {e}\n{traceback.format_exc()}")
            # Пытаемся вернуть последнее наблюдение
            try: last_obs = self._get_one_observation(self._current_time_step, current_player)
            except: last_obs = zero_obs_dict
            info["error"] = f"Step failed: {e}"
            return last_obs, 0.0, True, False, info # Завершаем эпизод при ошибке

    # --- Вспомогательные методы ---
    def _get_state(self): return getattr(self._base_env, '_state', None)
    def _get_pot_size(self, time_step): state = self._get_state(); return float(state.pot()) if state and hasattr(state, 'pot') else 0.0
    def _get_street(self, time_step): state = self._get_state(); return int(state.round()) if state and hasattr(state, 'round') else 0

    def _get_action_type(self, action):
        state = self._get_state(); current_player = self._current_time_step.observations["current_player"] if self._current_time_step else -1
        if state and hasattr(state, 'action_to_string') and current_player >= 0:
             try:
                 action_str = state.action_to_string(current_player, action).lower()
                 if "fold" in action_str: return 'fold'
                 if "check" in action_str: return 'call' # Check как разновидность Call(0)
                 if "call" in action_str: return 'call' # Call(amount > 0)
                 if "raise" in action_str or "bet" in action_str: return 'raise'
             except Exception: pass
        # Запасной вариант
        if action == 0: return 'fold';
        if action == 1: return 'call'; # Может быть check или call
        if action >= 2: return 'raise'; # Может быть bet или raise
        return 'unknown'

    def _calculate_pot_odds(self, time_step, player_id):
         state = self._get_state()
         if state and hasattr(state, 'amount_to_call') and hasattr(state, 'pot') and player_id >=0:
             try:
                 call_amount = max(0.0, float(state.amount_to_call(player_id)))
                 pot_size = float(state.pot())
                 total = pot_size + call_amount
                 odds = np.clip(call_amount / total, 0.0, 1.0) if total > 1e-6 else 0.0
                 return self.dtype(odds)
             except Exception: return self.dtype(0.0)
         return self.dtype(0.0)

    def _calculate_stack_to_pot(self, time_step, player_id):
         state = self._get_state()
         if state and hasattr(state, 'stacks') and hasattr(state, 'pot') and player_id >= 0:
             try:
                 stacks = state.stacks(); pot_size = float(state.pot())
                 if player_id < len(stacks):
                      stack = float(stacks[player_id])
                      # Ограничиваем SPR сверху для стабильности и нормализуем
                      spr = np.clip(stack / pot_size, 0.0, 100.0) if pot_size > 1e-6 else 100.0
                      return self.dtype(spr / 100.0)
                 else: return self.dtype(0.0)
             except Exception: return self.dtype(0.0)
         return self.dtype(0.0)

    def _enhance_info(self, info, current_player_id):
        """Добавляет полезную отладочную информацию в словарь info."""
        # Добавляем легальные действия (может быть полезно для отладки)
        if hasattr(self,'_current_time_step') and self._current_time_step and not self._current_time_step.last() and current_player_id >= 0:
             try: info['legal_actions'] = self._current_time_step.observations["legal_actions"].get(current_player_id, [])
             except Exception: info['legal_actions'] = []
        else: info['legal_actions'] = []
        # Добавляем другую информацию
        info['opponent_stats_features'] = {p: self.stats.get_features(p).tolist() for p in range(self.game.num_players())}
        info['pot_size'] = self.current_hand_info.get('pot_size', 0)
        info['street'] = self.current_street
        return info

    def render(self, mode='human'):
        """Выводит текущее состояние игры."""
        state = self._get_state()
        rendered = str(state) if state else "No state available for rendering."
        if mode == 'human': print(rendered)
        return rendered # Возвращает строковое представление для 'ansi'

    def close(self):
        """Освобождает ресурсы."""
        logger.info("Closing PokerEnv")
        self._base_env = None
        self.game = None

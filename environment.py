# environment.py
import numpy as np
import gymnasium as gym
from gymnasium.utils import seeding # Для np_random
from open_spiel.python import rl_environment
import pyspiel
from collections import deque
import logging
from typing import Dict, Any, List, Tuple
import traceback # Добавлено для более детального логирования ошибок

# Настройка логирования
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # Раскомментируй для детальной отладки
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO) # Установите уровень по умолчанию

class OpponentStats:
    """Класс для хранения и обновления базовой статистики оппонентов."""
    def __init__(self, num_players, dtype=np.float32): # Добавим dtype
        self.num_players = num_players
        self.stats = {}
        self.feature_size = 4 # VPIP, PFR, AF, Hands Played
        self.dtype = dtype
        self.reset()

    def reset(self):
        """Сбрасывает статистику для всех игроков."""
        self.stats = {}
        for i in range(self.num_players):
             self._ensure_player_stats(i)

    def _ensure_player_stats(self, player_id):
         """Гарантирует наличие записи для игрока."""
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

        # Расчет VPIP/PFR/AF на лету
        hands = player_stats.get('hands', 0)
        vpip_opps = player_stats.get('vpip_opportunities', 0)
        pfr_opps = player_stats.get('pfr_opportunities', 0)
        vpip_actions = player_stats.get('vpip_actions', 0)
        pfr_actions = player_stats.get('pfr_actions', 0)
        agg_actions = player_stats.get('agg_actions', 0)
        pass_actions = player_stats.get('pass_actions', 0)

        vpip = float(vpip_actions / vpip_opps) if vpip_opps > 0 else 0.0
        pfr = float(pfr_actions / pfr_opps) if pfr_opps > 0 else 0.0

        if pass_actions > 0:
            af = float(agg_actions / pass_actions)
        elif agg_actions > 0:
            af = 5.0 # Условное высокое значение
        else:
            af = 1.0 # Нейтральное значение по умолчанию

        # Нормализация (примерная)
        norm_vpip = np.clip(vpip, 0.0, 1.0)
        norm_pfr = np.clip(pfr, 0.0, 1.0)
        norm_af = np.clip(af / 5.0, 0.0, 1.0) # Масштабируем AF к ~[0,1]
        norm_hands = np.clip(hands / 100.0, 0.0, 1.0) # Нормализуем количество рук (до 100)

        # --- ИСПРАВЛЕНО: Явное указание dtype ---
        return np.array([norm_vpip, norm_pfr, norm_af, norm_hands], dtype=self.dtype)

    def record_opportunity(self, player_id, is_vpip_opp, is_pfr_opp):
        """Записывает возможность для VPIP/PFR."""
        player_id_int = int(player_id)
        self._ensure_player_stats(player_id_int)
        if is_vpip_opp:
             self.stats[player_id_int]['vpip_opportunities'] += 1
        if is_pfr_opp:
             self.stats[player_id_int]['pfr_opportunities'] += 1

    def update_on_action(self, player_id, action_type, street, is_voluntary_action):
        """Обновляет счетчики действий."""
        player_id_int = int(player_id)
        self._ensure_player_stats(player_id_int)
        stats = self.stats[player_id_int]

        if street == 0 and is_voluntary_action and not stats['vpip_acted_this_hand']:
             stats['vpip_actions'] += 1
             stats['vpip_acted_this_hand'] = True

        if street == 0 and action_type == 'raise' and not stats['pfr_acted_this_hand']:
             stats['pfr_actions'] += 1
             stats['pfr_acted_this_hand'] = True

        if street > 0:
            if action_type == 'raise':
                stats['agg_actions'] += 1
            elif action_type == 'call':
                stats['pass_actions'] += 1

    def finalize_hand_stats(self, involved_players):
         """Вызывается в конце руки."""
         for player_id in involved_players:
             player_id_int = int(player_id)
             self._ensure_player_stats(player_id_int)
             self.stats[player_id_int]['hands'] += 1
             self.stats[player_id_int]['vpip_acted_this_hand'] = False
             self.stats[player_id_int]['pfr_acted_this_hand'] = False


class PokerEnv(gym.Env):
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}

    def __init__(self, env_config: dict):
        super().__init__()
        logger.info("Initializing PokerEnv...")

        # Проверка и извлечение конфига
        if "config" not in env_config:
             if "env_config" in env_config:
                 env_config = env_config["env_config"]
             else:
                 raise ValueError("Missing 'config' key (PokerConfig instance) in env_config dictionary.")
        if "config" not in env_config:
            raise ValueError("Missing 'config' key (PokerConfig instance) after checking nested 'env_config'.")

        self.config = env_config["config"]
        # --- ИСПРАВЛЕНО: Получаем dtype из env_config или используем float32 по умолчанию ---
        self.dtype = env_config.get("dtype", np.float32)
        if not isinstance(self.dtype, type) or not np.issubdtype(self.dtype, np.floating):
             logger.warning(f"Invalid dtype specified: {self.dtype}. Defaulting to np.float32.")
             self.dtype = np.float32

        # --- OpenSpiel Initialization ---
        game_params_from_config = self.config.game_config if hasattr(self.config, 'game_config') else {}
        game_name = self.config.game_name if hasattr(self.config, 'game_name') else "leduc_poker"

        processed_game_params = {}
        for k, v in game_params_from_config.items():
            if k in ["numBoardCards", "blind", "stack"] and isinstance(v, (list, tuple, np.ndarray)):
                processed_game_params[k] = " ".join(map(str, v))
            elif isinstance(v, bool):
                 processed_game_params[k] = str(v).lower()
            elif isinstance(v, (int, float, np.number)):
                 processed_game_params[k] = v
            else:
                 processed_game_params[k] = str(v)

        logger.info(f"Attempting to load OpenSpiel game: '{game_name}' with processed params: {processed_game_params}")
        try:
            self.game = pyspiel.load_game(game_name, processed_game_params)
            logger.info(f"Game '{game_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load game '{game_name}' with params: {str(processed_game_params)} - Error: {e}", exc_info=True)
            raise

        try:
            self._base_env = rl_environment.Environment(game=self.game, include_full_state=False)
            logger.info("OpenSpiel RL Environment created successfully.")
            self._action_spec = self._base_env.action_spec()
            self._observation_spec = self._base_env.observation_spec()
        except Exception as e:
            logger.error(f"Failed to create OpenSpiel RL Environment: {e}", exc_info=True)
            raise

        num_players_actual = self.game.num_players()
        # --- ИСПРАВЛЕНО: Передаем dtype в OpponentStats ---
        self.stats = OpponentStats(num_players_actual, dtype=self.dtype)

        # --- ИСПРАВЛЕНО: Логика определения пространств ---
        self._define_spaces(num_players_actual)

        self._position_names = ['SB', 'BB', 'UTG', 'MP1', 'MP2', 'MP3', 'CO', 'BTN'][:num_players_actual]
        self.current_street = 0
        self._current_time_step = None
        # Инициализация генератора случайных чисел Gymnasium
        self.np_random = None
        self.seed() # Инициализируем np_random

        logger.info(f"PokerEnv initialized for {num_players_actual} players. Obs space: {self.observation_space}, Action space: {self.action_space}")

    def seed(self, seed=None):
        """Устанавливает seed для генератора случайных чисел."""
        self.np_random, seed = seeding.np_random(seed)
        # OpenSpiel среда сама управляет своим RNG, но мы можем иметь свой для env
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
        # self.stats.reset() # Сброс статов между руками, если нужно

    def _define_spaces(self, num_players):
        """Определяет action_space и observation_space для Gymnasium."""
        # 1. Пространство действий
        num_actions = self._action_spec["num_actions"]
        self.action_space = gym.spaces.Discrete(num_actions)
        logger.info(f"Action space defined: Discrete({self.action_space.n})")

        # 2. Пространство наблюдений
        try:
            # --- ИСПРАВЛЕННЫЙ СПОСОБ ОПРЕДЕЛЕНИЯ РАЗМЕРА OBS ---
            # Получаем РЕАЛЬНЫЙ размер базового наблюдения от OpenSpiel
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

            # Общий размер вектора наблюдения
            total_obs_size = base_obs_size + num_extra_features
            logger.info(f"Calculated observation space size: {total_obs_size} (Base: {base_obs_size}, Extra: {num_extra_features})")

            # Определяем пространство Box
            # --- ИСПРАВЛЕНО: Используем -inf, inf для большей гибкости ---
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(total_obs_size,), dtype=self.dtype
            )
            logger.info(f"Observation space defined: {self.observation_space}")

        except Exception as e:
             logger.error(f"CRITICAL: Failed to determine observation space size: {e}. Cannot proceed.", exc_info=True)
             # Если не удалось определить размер, среда неработоспособна
             raise RuntimeError(f"Failed to initialize observation space: {e}")

    def _get_one_observation(self, time_step, player_id):
         """Собирает полный вектор наблюдения для заданного игрока."""
         # Проверка, что пространство было определено (должно быть после _define_spaces)
         if not hasattr(self, 'observation_space') or self.observation_space is None:
             # Эта ситуация не должна возникать при правильной инициализации
             logger.error("PANIC: _get_one_observation called before observation_space was defined!")
             raise RuntimeError("Observation space is not initialized before call.")

         obs_shape = self.observation_space.shape
         fallback_obs = np.zeros(obs_shape, dtype=self.dtype)

         if player_id < 0 or player_id >= self.game.num_players():
             logger.warning(f"Invalid player_id ({player_id}) requested in _get_one_observation. Returning zeros.")
             return fallback_obs

         try:
            # 1. Базовый вектор наблюдения (info_state) от OpenSpiel
            base_obs_list = time_step.observations["info_state"][player_id]
            # --- ИСПРАВЛЕНО: Сразу создаем с нужным dtype ---
            base_obs = np.array(base_obs_list, dtype=self.dtype)

            # 2. Фича позиции игрока
            position_feature = np.array([float(player_id) / max(1, self.game.num_players() - 1)], dtype=self.dtype)

            # 3. Фичи статистики оппонентов
            opponent_features_list = []
            for pid in range(self.game.num_players()):
                if pid != player_id:
                    opponent_features_list.extend(self.stats.get_features(pid)) # get_features уже возвращает нужный dtype
            opponent_features = np.array(opponent_features_list, dtype=self.dtype)
            # Паддинг/Обрезка на всякий случай (хотя не должны требоваться при правильной логике)
            expected_opp_len = (self.game.num_players() - 1) * self.stats.feature_size
            if len(opponent_features) != expected_opp_len:
                 logger.warning(f"Opponent features length mismatch! Got {len(opponent_features)}, expected {expected_opp_len}. Padding/truncating.")
                 opponent_features = np.resize(opponent_features, expected_opp_len).astype(self.dtype)


            # 4. Фича шансов банка (Pot Odds)
            pot_odds_feature = np.array([self._calculate_pot_odds(time_step, player_id)], dtype=self.dtype)

            # 5. Фича отношения стека к поту (SPR)
            stack_to_pot_feature = np.array([self._calculate_stack_to_pot(time_step, player_id)], dtype=self.dtype)

            # --- СБОРКА Финального Вектора ---
            # --- УБРАНА ПРОВЕРКА if len(base_obs) != expected_base_len: ---
            # Эта проверка вызывала ошибки из-за неправильного расчета expected_base_len
            # Теперь мы полагаемся на то, что _define_spaces правильно вычислил общий размер.

            # Собираем все части
            full_obs_parts = [
                base_obs,
                position_feature,
                pot_odds_feature,
                stack_to_pot_feature,
                opponent_features
            ]
            full_obs = np.concatenate(full_obs_parts).astype(self.dtype) # Финальное приведение типа на всякий случай

            # Проверка финальной длины (для отладки)
            if len(full_obs) != obs_shape[0]:
                 logger.error(
                     f"FINAL observation length mismatch! Expected {obs_shape[0]}, got {len(full_obs)}. "
                     f"Base: {len(base_obs)}, Pos: {len(position_feature)}, PO: {len(pot_odds_feature)}, "
                     f"SPR: {len(stack_to_pot_feature)}, OppStats: {len(opponent_features)}. Returning zeros."
                 )
                 return fallback_obs

            # Клиппинг, если границы [-1, 1] (сейчас не используется по умолчанию)
            # if self.observation_space.low.min() >= -1 and self.observation_space.high.max() <= 1:
            #     full_obs = np.clip(full_obs, -1.0, 1.0)

            return full_obs

         except Exception as e:
            logger.error(f"Error constructing observation for player {player_id}: {e}\n{traceback.format_exc()}")
            return fallback_obs # Возвращаем нули в случае любой другой ошибки

    def reset(self, seed=None, options=None):
        # --- ИСПРАВЛЕНО: Инициализация RNG через super().reset() ---
        super().reset(seed=seed)
        logger.debug("Resetting environment...")
        try:
            self._current_time_step = self._base_env.reset()
            self.current_street = 0
            self._reset_metrics()

            # Упрощенная запись возможностей VPIP/PFR
            is_vpip_opp = True
            is_pfr_opp = True
            for p in range(self.game.num_players()):
                self.stats.record_opportunity(p, is_vpip_opp, is_pfr_opp)

            while self._current_time_step.observations["current_player"] == pyspiel.PlayerId.CHANCE:
                 logger.debug("Handling initial chance node.")
                 self._current_time_step = self._base_env.step([])

            current_player = self._current_time_step.observations["current_player"]
            if self._current_time_step.last() or current_player < 0:
                logger.warning("Environment terminated immediately after reset. Returning zero observation.")
                obs = np.zeros(self.observation_space.shape, dtype=self.dtype)
                info = {}
                return obs, info

            obs = self._get_one_observation(self._current_time_step, current_player)
            info = self._enhance_info({}, current_player)

            logger.debug(f"Reset complete. First player: {current_player}")
            # --- ИСПРАВЛЕНО: Гарантируем dtype на выходе reset ---
            return obs.astype(self.dtype), info

        except Exception as e:
            logger.error(f"Error during environment reset: {e}\n{traceback.format_exc()}")
            obs = np.zeros(self.observation_space.shape, dtype=self.dtype)
            info = {"error": f"Reset failed: {e}"}
            return obs, info

    def step(self, action):
        # Преобразование действия
        action = int(action.item()) if hasattr(action, 'item') else int(action)

        if self._current_time_step is None or self._current_time_step.last():
            logger.warning("Step called on a terminal or invalid state. Resetting environment.")
            obs, info = self.reset()
            # Возвращаем состояние после сброса, но с нулевой наградой и terminated=True
            return obs, 0.0, True, False, info

        current_player = self._current_time_step.observations["current_player"]
        if current_player < 0 :
             logger.error(f"Invalid current player ID ({current_player}) in step. Resetting.")
             obs, info = self.reset()
             return obs, 0.0, True, False, info

        logger.debug(f"Player {current_player} attempts action: {action}")

        legal_actions = self._current_time_step.observations["legal_actions"][current_player]
        if not legal_actions:
             logger.warning(f"Player {current_player} has no legal actions. Ending episode.")
             terminated = True
             truncated = False
             reward = float(self._current_time_step.rewards[current_player]) if self._current_time_step.rewards else 0.0
             obs = self._get_one_observation(self._current_time_step, current_player) # Последнее наблюдение
             info = self._enhance_info({'error': 'No legal actions'}, current_player)
             involved_players = self.current_hand_info.get('involved_players', set(range(self.game.num_players())))
             self.stats.finalize_hand_stats(involved_players)
             self._current_time_step = None # Сбрасываем, т.к. эпизод завершен
             return obs, reward, terminated, truncated, info

        if action not in legal_actions:
            original_action = action
            # --- ИСПРАВЛЕНО: Используем self.np_random ---
            action = self.np_random.choice(legal_actions)
            logger.warning(f"Player {current_player} illegal action {original_action}. Corrected to {action}. Legal: {legal_actions}")

        try:
            # Обновление статистики и информации о руке ДО шага
            action_type = self._get_action_type(action)
            is_voluntary = (action_type != 'fold')
            self.stats.update_on_action(current_player, action_type, self.current_street, is_voluntary)
            self.current_hand_info['actions_by_player'][current_player].append(action)
            if action_type != 'fold':
                self.current_hand_info['involved_players'].add(current_player)
            if action_type == 'raise':
                 self.current_hand_info['last_raiser'] = current_player

            # Шаг в среде OpenSpiel
            self._current_time_step = self._base_env.step([action])

            # Обновление информации ПОСЛЕ шага
            self.current_hand_info['pot_size'] = self._get_pot_size(self._current_time_step)
            next_player = self._current_time_step.observations["current_player"]
            self.current_hand_info['current_player'] = next_player

            # Обработка Chance узлов
            while self._current_time_step.observations["current_player"] == pyspiel.PlayerId.CHANCE:
                 logger.debug("Handling chance node.")
                 self._current_time_step = self._base_env.step([])
                 next_player = self._current_time_step.observations["current_player"]

            # Обновление улицы
            new_street = self._get_street(self._current_time_step)
            if new_street != self.current_street:
                logger.debug(f"Street changed from {self.current_street} to {new_street}")
                self.current_street = new_street

            # Получение результатов шага
            reward = float(self._current_time_step.rewards[current_player]) if self._current_time_step.rewards else 0.0
            terminated = self._current_time_step.last()
            truncated = False # У нас нет усечения по времени

            # Получение наблюдения для следующего игрока
            obs_player_id = next_player if not terminated else current_player
            if obs_player_id < 0: obs_player_id = current_player # Если next_player == TERMINAL
            obs = self._get_one_observation(self._current_time_step, obs_player_id)

            # Обновление наград эпизода
            self.episode_rewards[current_player] += reward

            info = self._enhance_info({}, obs_player_id)
            if terminated:
                logger.debug(f"Episode terminated. Final rewards: {self._current_time_step.rewards}")
                involved_players = self.current_hand_info.get('involved_players', set(range(self.game.num_players())))
                self.stats.finalize_hand_stats(involved_players)
                # Записываем финальные награды для всех (они могут отличаться от награды последнего шага)
                final_rewards = {p: float(r) for p, r in enumerate(self._current_time_step.rewards)}
                self.episode_rewards = final_rewards # Перезаписываем последними наградами
                info['episode_rewards'] = final_rewards
                logger.info(f"Episode finished. Total rewards: {self.episode_rewards}")

            # --- ИСПРАВЛЕНО: Гарантируем dtype на выходе step ---
            return obs.astype(self.dtype), reward, terminated, truncated, info

        except Exception as e:
            logger.error(f"Error during environment step: {e}\n{traceback.format_exc()}")
            obs = np.zeros(self.observation_space.shape, dtype=self.dtype)
            info = {"error": f"Step failed: {e}"}
            # Завершаем эпизод при ошибке
            return obs, 0.0, True, False, info

    # --- Вспомогательные методы (без изменений, но проверяем dtype) ---

    def _get_state(self):
        return getattr(self._base_env, '_state', None)

    def _get_pot_size(self, time_step):
        state = self._get_state()
        if state and hasattr(state, 'pot'):
            try: return float(state.pot())
            except Exception: return 0.0
        return 0.0

    def _get_action_type(self, action):
        state = self._get_state()
        current_player = self._current_time_step.observations["current_player"] if self._current_time_step else -1
        if state and hasattr(state, 'action_to_string') and current_player >= 0:
             try:
                 action_str = state.action_to_string(current_player, action).lower()
                 if "fold" in action_str: return 'fold'
                 if "check" in action_str or ("call" in action_str and action == 1): return 'call' # Уточнение для check/call
                 if "raise" in action_str or "bet" in action_str or ("call" in action_str and action > 1) : return 'raise' # Включаем call(amount>0)
             except Exception as e:
                 logger.warning(f"Error calling action_to_string for action {action}: {e}. Defaulting.")
        # Запасной вариант (менее надежный)
        if action == 0: return 'fold'
        if action == 1: return 'call' # Может быть check
        if action >= 2: return 'raise'
        return 'unknown'

    def _get_street(self, time_step):
        state = self._get_state()
        if state and hasattr(state, 'round'):
            try: return int(state.round())
            except Exception: return 0
        return 0

    def _calculate_pot_odds(self, time_step, player_id):
         state = self._get_state()
         if state and hasattr(state, 'amount_to_call') and hasattr(state, 'pot') and player_id >=0:
             try:
                 call_amount = float(state.amount_to_call(player_id))
                 if call_amount < 0: call_amount = 0
                 pot_size = float(state.pot())
                 total_pot_after_call = pot_size + call_amount
                 if total_pot_after_call > 1e-6:
                     pot_odds = np.clip(call_amount / total_pot_after_call, 0.0, 1.0)
                     # --- ИСПРАВЛЕНО: Возвращаем нужный dtype ---
                     return self.dtype(pot_odds)
                 else: return self.dtype(0.0)
             except Exception: return self.dtype(0.0)
         return self.dtype(0.0)

    def _calculate_stack_to_pot(self, time_step, player_id):
         state = self._get_state()
         if state and hasattr(state, 'stacks') and hasattr(state, 'pot') and player_id >= 0:
             try:
                 stacks = state.stacks()
                 pot_size = float(state.pot())
                 if player_id < len(stacks):
                      player_stack = float(stacks[player_id])
                      if pot_size > 1e-6:
                          spr = np.clip(player_stack / pot_size, 0.0, 100.0) # Ограничение SPR
                          # --- ИСПРАВЛЕНО: Возвращаем нужный dtype ---
                          return self.dtype(spr / 100.0) # Нормализация к [0, 1]
                      else: return self.dtype(1.0) # Макс. нормализованное значение, если пот 0
                 else: return self.dtype(0.0)
             except Exception: return self.dtype(0.0)
         return self.dtype(0.0)

    def _enhance_info(self, info, current_player_id):
        """Добавляет полезную отладочную информацию в словарь info."""
        if hasattr(self,'_current_time_step') and self._current_time_step and not self._current_time_step.last() and current_player_id >= 0:
             try:
                 if current_player_id < len(self._current_time_step.observations["legal_actions"]):
                     info['legal_actions'] = self._current_time_step.observations["legal_actions"][current_player_id]
                 else: info['legal_actions'] = []
             except Exception: info['legal_actions'] = []
        else: info['legal_actions'] = []

        info['opponent_stats_features'] = {p: self.stats.get_features(p).tolist() for p in range(self.game.num_players())}
        info['pot_size'] = self.current_hand_info.get('pot_size', 0)
        info['street'] = self.current_street
        return info

    def render(self, mode='human'):
        """Выводит текущее состояние игры."""
        state = self._get_state()
        rendered = str(state) if state else "No state available."
        if mode == 'human': print(rendered)
        return rendered

    def close(self):
        """Освобождает ресурсы."""
        logger.info("Closing PokerEnv")
        self._base_env = None
        self.game = None

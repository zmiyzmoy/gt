# environment.py
import numpy as np
import gymnasium as gym
from open_spiel.python import rl_environment
import pyspiel
from collections import deque
import logging
from typing import Dict, Any, List, Tuple

# Настройка логирования
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # Раскомментируй для детальной отладки

class OpponentStats:
    """Класс для хранения и обновления базовой статистики оппонентов."""
    def __init__(self, num_players):
        self.num_players = num_players
        self.stats = {}
        self.feature_size = 4 # VPIP, PFR, AF, Hands
        self.reset()

    def reset(self):
        self.stats = {}
        for i in range(self.num_players):
             self._ensure_player_stats(i)

    def _ensure_player_stats(self, player_id):
         player_id_int = int(player_id)
         if player_id_int not in self.stats:
             self.stats[player_id_int] = {'vpip': 0.0, 'pfr': 0.0, 'af': 1.0, 'hands': 0,
                                         'agg_actions': 0, 'pass_actions': 0,
                                         'vpip_opportunities': 0, 'pfr_opportunities': 0,
                                         'vpip_acted_this_hand': False, 'pfr_acted_this_hand': False}

    def get_features(self, player_id):
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
        if pass_actions > 0: af = float(agg_actions / pass_actions)
        elif agg_actions > 0: af = 5.0
        else: af = 1.0
        norm_af = min(af / 5.0, 1.0)
        norm_hands = min(hands / 100.0, 1.0)
        return np.array([vpip, pfr, norm_af, norm_hands], dtype=np.float32)

    def record_opportunity(self, player_id, is_vpip_opp, is_pfr_opp):
        player_id_int = int(player_id)
        self._ensure_player_stats(player_id_int)
        if is_vpip_opp:
             self.stats[player_id_int]['vpip_opportunities'] = self.stats[player_id_int].get('vpip_opportunities', 0) + 1
             self.stats[player_id_int]['vpip_acted_this_hand'] = False
        if is_pfr_opp:
             self.stats[player_id_int]['pfr_opportunities'] = self.stats[player_id_int].get('pfr_opportunities', 0) + 1
             self.stats[player_id_int]['pfr_acted_this_hand'] = False

    def update_on_action(self, player_id, action_type, street, is_voluntary_action):
        player_id_int = int(player_id)
        self._ensure_player_stats(player_id_int)
        stats = self.stats[player_id_int]
        if street == 0 and is_voluntary_action and not stats['vpip_acted_this_hand']:
             stats['vpip_actions'] = stats.get('vpip_actions', 0) + 1
             stats['vpip_acted_this_hand'] = True
        if street == 0 and action_type == 'raise' and not stats['pfr_acted_this_hand']:
             stats['pfr_actions'] = stats.get('pfr_actions', 0) + 1
             stats['pfr_acted_this_hand'] = True
        if street > 0:
            if action_type == 'raise': stats['agg_actions'] = stats.get('agg_actions', 0) + 1
            elif action_type == 'call': stats['pass_actions'] = stats.get('pass_actions', 0) + 1

    def finalize_hand_stats(self, involved_players):
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

        if "config" not in env_config:
             raise ValueError("Missing 'config' key in env_config dictionary.")
        self.config = env_config["config"] # Экземпляр PokerConfig
        self.dtype = env_config.get("dtype", np.float32)

        # --- OpenSpiel Initialization ---
        # --- ИЗМЕНЕНО: Конвертируем ВСЕ значения в СТРОКИ для OpenSpiel 1.3 ---
        game_params_from_config = self.config.game_config
        processed_game_params = {}
        for k, v in game_params_from_config.items():
             if isinstance(v, (list, tuple, np.ndarray)):
                 processed_game_params[k] = " ".join(map(str, v)) # Списки -> строки
             else:
                 processed_game_params[k] = str(v) # Все остальное -> строки
        # --------------------------------------------------------------------

        logger.info(f"Creating OpenSpiel game: {self.config.game_name} with params: {processed_game_params}")
        try:
            self.game = pyspiel.load_game(self.config.game_name, processed_game_params)
            logger.info("Game loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load game '{self.config.game_name}' with params {processed_game_params}: {e}", exc_info=True)
            raise

        try:
            self._base_env = rl_environment.Environment(game=self.game, include_full_state=False)
            logger.info("OpenSpiel RL Environment created successfully")
            self._action_spec = self._base_env.action_spec()
            self._observation_spec = self._base_env.observation_spec()
        except Exception as e:
            logger.error(f"Failed to create OpenSpiel RL Environment: {e}", exc_info=True)
            raise

        self.stats = OpponentStats(self.config.num_players)
        self._define_spaces()

        self._position_names = ['SB', 'BB', 'UTG', 'MP1', 'MP2', 'MP3', 'CO', 'BTN'][:self.config.num_players]
        self.current_street = 0
        self._current_time_step = None

        logger.info(f"PokerEnv initialized for {self.config.num_players} players.")

    def _reset_metrics(self):
        self.episode_rewards = {p: 0.0 for p in range(self.config.num_players)}
        self.current_hand_info = {
            'actions_by_player': {p: [] for p in range(self.config.num_players)},
            'involved_players': set(),
            'pot_size': 0,
            'current_player': -1,
            'last_raiser': -1,
            'can_check': {}
        }

    def _define_spaces(self):
        num_actions = self._action_spec["num_actions"]
        self.action_space = gym.spaces.Discrete(num_actions)
        logger.info(f"Action space defined: Discrete({self.action_space.n})")

        try:
            time_step = self._base_env.reset()
            player_id = time_step.observations["current_player"]
            if player_id < 0 and len(time_step.observations["info_state"]) > 0: player_id = 0
            if player_id >= 0:
                 processed_obs_vector = self._get_one_observation(time_step, player_id)
                 obs_size = len(processed_obs_vector)
                 logger.info(f"Calculated Observation space size: {obs_size}")
                 self.observation_space = gym.spaces.Box(
                     low=-np.inf, high=np.inf, shape=(obs_size,), dtype=self.dtype
                 )
            else: raise ValueError("Could not determine initial player.")
        except Exception as e:
             logger.error(f"Failed to determine observation size: {e}. Using fallback 512.", exc_info=True)
             self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(512,), dtype=self.dtype)

    def _get_one_observation(self, time_step, player_id):
         if player_id < 0 or player_id >= len(time_step.observations["info_state"]):
             shape_to_use = self.observation_space.shape if hasattr(self, 'observation_space') else (512,)
             return np.zeros(shape_to_use, dtype=self.dtype)
         try:
            base_obs_list = time_step.observations["info_state"][player_id]
            base_obs = np.array(base_obs_list, dtype=self.dtype) # Используем self.dtype (float32)

            position_feature = np.array([float(player_id) / max(1, self.config.num_players - 1)], dtype=self.dtype)

            opponent_features_list = []
            for pid in range(self.config.num_players):
                if pid != player_id: opponent_features_list.extend(self.stats.get_features(pid))
            opponent_features = np.array(opponent_features_list, dtype=self.dtype)

            pot_odds_feature = np.array([self._calculate_pot_odds(time_step, player_id)], dtype=self.dtype)
            stack_to_pot_feature = np.array([self._calculate_stack_to_pot(time_step, player_id)], dtype=self.dtype)

            full_obs = np.concatenate([
                base_obs, position_feature, pot_odds_feature,
                stack_to_pot_feature, opponent_features
            ]).astype(self.dtype) # Гарантируем тип

            expected_len = self.observation_space.shape[0]
            if len(full_obs) != expected_len:
                 logger.warning(f"Obs length mismatch! Got {len(full_obs)}, expected {expected_len}. Padding/truncating.")
                 final_obs = np.zeros(expected_len, dtype=self.dtype)
                 size = min(len(full_obs), expected_len)
                 final_obs[:size] = full_obs[:size]
                 return final_obs
            return full_obs
         except Exception as e:
            logger.error(f"Error in _get_one_observation for player {player_id}: {e}", exc_info=True)
            return np.zeros(self.observation_space.shape, dtype=self.dtype)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        logger.info("Resetting environment")
        try:
            self._current_time_step = self._base_env.reset()
            self.current_street = 0
            self._reset_metrics()
            for p in range(self.config.num_players):
                 is_vpip_opp = (p != 1); is_pfr_opp = True
                 self.stats.record_opportunity(p, is_vpip_opp, is_pfr_opp)
            player_id = self._current_time_step.observations["current_player"]
            if player_id < 0: player_id = 0
            obs_vector = self._get_one_observation(self._current_time_step, player_id)
            return obs_vector, {}
        except Exception as e:
            logger.error(f"Error during reset: {e}", exc_info=True)
            return np.zeros(self.observation_space.shape, dtype=self.dtype), {}

    def step(self, action):
        if isinstance(action, np.generic): action = int(action)
        player_before_action = self._current_time_step.observations["current_player"] if self._current_time_step else -1
        logger.debug(f"Player {player_before_action} received action: {action}")

        if self._current_time_step is None or self._current_time_step.last():
             logger.warning("Step on terminated/uninitialized env. Resetting.")
             obs, info = self.reset()
             return obs, 0.0, True, False, info

        current_player = self._current_time_step.observations["current_player"]
        legal_actions = self._current_time_step.observations["legal_actions"][current_player]

        if action not in legal_actions:
            logger.warning(f"Illegal action {action} by p{current_player}. Legal: {legal_actions}.")
            action = self.np_random.choice(legal_actions) if legal_actions else 0
            logger.warning(f"Replaced with random legal action: {action}")

        try:
            action_type = self._get_action_type(action)
            is_voluntary = (self.current_street == 0 and action_type != 'fold') or \
                           (self.current_street > 0 and action_type != 'fold')
            self.stats.update_on_action(current_player, action_type, self.current_street, is_voluntary)
            self.current_hand_info['actions_by_player'][current_player].append(action)
            self.current_hand_info['involved_players'].add(current_player)

            self._current_time_step = self._base_env.step([action])

            self.current_hand_info['pot_size'] = self._get_pot_size(self._current_time_step)
            next_player = self._current_time_step.observations["current_player"]
            self.current_hand_info['current_player'] = next_player

            new_street = self._get_street(self._current_time_step)
            if new_street != self.current_street:
                self.current_street = new_street

            reward = float(self._current_time_step.rewards[current_player])
            terminated = self._current_time_step.last()
            truncated = False
            obs_vector = self._get_one_observation(self._current_time_step, next_player)

            if terminated:
                self.stats.finalize_hand_stats(self.current_hand_info['involved_players'])
                for p, r in enumerate(self._current_time_step.rewards):
                     self.episode_rewards[p] = self.episode_rewards.get(p, 0.0) + float(r)

            info = self._enhance_info({}, next_player)
            return obs_vector, reward, terminated, truncated, info

        except Exception as e:
            logger.error(f"Error during step: {e}", exc_info=True)
            return np.zeros(self.observation_space.shape, dtype=self.dtype), 0.0, True, False, {"error": str(e)}

    def _get_state(self): return getattr(self._base_env, '_state', None)
    def _get_pot_size(self, time_step):
        state = self._get_state(); return float(state.pot()) if state and hasattr(state, 'pot') else 0.0
    def _get_action_type(self, action):
        state = self._get_state(); current_player = self._current_time_step.observations["current_player"] if self._current_time_step else -1
        if state and hasattr(state, 'action_to_string') and current_player >= 0:
             try:
                 action_str = state.action_to_string(current_player, action)
                 if "fold" in action_str.lower(): return 'fold'
                 if "check" in action_str.lower() or "call" in action_str.lower(): return 'call'
                 if "raise" in action_str.lower() or "bet" in action_str.lower(): return 'raise'
             except: pass
        return 'fold' if action == 0 else ('call' if action == 1 else 'raise')
    def _get_street(self, time_step): state = self._get_state(); return int(state.round()) if state and hasattr(state, 'round') else 0
    def _calculate_pot_odds(self, time_step, player_id):
         state = self._get_state()
         if state and hasattr(state, 'amount_to_call') and hasattr(state, 'pot') and player_id >=0:
             try:
                 call_amount = float(state.amount_to_call(player_id)); pot_size = float(state.pot())
                 total_pot = pot_size + call_amount
                 return min(call_amount / total_pot, 1.0) if total_pot > 1e-6 else 0.0
             except: pass
         return 0.0
    def _calculate_stack_to_pot(self, time_step, player_id):
         state = self._get_state()
         if state and hasattr(state, 'stacks') and hasattr(state, 'pot') and player_id >= 0:
             try:
                 stacks = state.stacks()
                 if player_id < len(stacks):
                      player_stack = float(stacks[player_id]); pot_size = float(state.pot())
                      return min(player_stack / pot_size, 100.0) if pot_size > 1e-6 else 100.0
             except: pass
         return 100.0
    def _enhance_info(self, info, next_player_id):
        if not self._current_time_step.last() and next_player_id >= 0:
             try: info['legal_actions'] = self._current_time_step.observations["legal_actions"][next_player_id]
             except: info['legal_actions'] = []
        info['metrics'] = {p: self.stats.get_features(p).tolist() for p in range(self.config.num_players)}
        return info
    def render(self, mode='human'): state = self._get_state(); rendered = str(state) if state else "No state."; print(rendered); return rendered
    def close(self): logger.info("Closing PokerEnv"); self._base_env = None; self.game = None

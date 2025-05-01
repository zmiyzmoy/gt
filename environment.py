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
    # ... (код OpponentStats остается БЕЗ ИЗМЕНЕНИЙ) ...
    def __init__(self, num_players): self.num_players = num_players; self.stats = {}; self.feature_size = 4; self.reset()
    def reset(self): self.stats = {}; [self._ensure_player_stats(i) for i in range(self.num_players)]
    def _ensure_player_stats(self, i): i=int(i); self.stats.setdefault(i,{'vpip':0,'pfr':0,'af':1,'hands':0,'agg':0,'pass':0,'vpip_opp':0,'pfr_opp':0,'vpip_act_count':0,'pfr_act_count':0,'vpip_acted':False,'pfr_acted':False}) # Добавил _count
    def get_features(self, i): i=int(i); self._ensure_player_stats(i); s=self.stats[i]; h=s.get('hands',0); vo=s.get('vpip_opp',0); po=s.get('pfr_opp',0); va=s.get('vpip_act_count',0); pa=s.get('pfr_act_count',0); ag=s.get('agg',0); ps=s.get('pass',0); vpip=float(va/vo)if vo>0 else 0; pfr=float(pa/po)if po>0 else 0; af=float(ag/ps)if ps>0 else(5 if ag>0 else 1); naf=min(af/5,1); nh=min(h/100,1); return np.array([vpip,pfr,naf,nh],dtype=np.float32)
    def record_opportunity(self, i, v, p): i=int(i); self._ensure_player_stats(i); if v: self.stats[i]['vpip_opp']=self.stats[i].get('vpip_opp',0)+1; self.stats[i]['vpip_acted']=False; if p: self.stats[i]['pfr_opp']=self.stats[i].get('pfr_opp',0)+1; self.stats[i]['pfr_acted']=False
    def update_on_action(self, i, t, s, vol): i=int(i); self._ensure_player_stats(i); st=self.stats[i]; if s==0 and vol and not st['vpip_acted']: st['vpip_act_count']=st.get('vpip_act_count',0)+1; st['vpip_acted']=True; if s==0 and t=='raise' and not st['pfr_acted']: st['pfr_act_count']=st.get('pfr_act_count',0)+1; st['pfr_acted']=True; if s>0: (st['agg']:=st.get('agg',0)+1) if t=='raise' else (st['pass']:=st.get('pass',0)+1) if t=='call' else None
    def finalize_hand_stats(self, inv): [ (self._ensure_player_stats(int(p)), self.stats[int(p)].update({'hands': self.stats[int(p)].get('hands',0)+1, 'vpip_acted': False, 'pfr_acted': False})) for p in inv]

class PokerEnv(gym.Env):
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}

    def __init__(self, env_config: dict):
        super().__init__()
        logger.info("Initializing PokerEnv...")

        if "config" not in env_config:
             raise ValueError("Missing 'config' key in env_config dictionary.")
        self.config = env_config["config"]
        self.dtype = env_config.get("dtype", np.float32)

        # --- OpenSpiel Initialization ---
        game_params_from_config = self.config.game_config
        # --- ИЗМЕНЕНО: Правильная конвертация типов для OpenSpiel 1.3 ---
        processed_game_params = {}
        for k, v in game_params_from_config.items():
            if k in ["numBoardCards", "blind"]: # Списки -> строки с пробелами
                processed_game_params[k] = " ".join(map(str, v))
            # --- Stack и другие ЧИСЛА остаются числами (int/float) ---
            elif isinstance(v, (int, float, np.number)):
                processed_game_params[k] = v # Оставляем как число!
            elif isinstance(v, bool):
                 processed_game_params[k] = str(v).lower() # Булевы -> "true"/"false"
            else:
                 # Все остальное (напр. betting) -> строки
                 processed_game_params[k] = str(v)
        # -----------------------------------------------------------------

        logger.info(f"Creating OpenSpiel game: {self.config.game_name} with FINAL params: {processed_game_params}")
        try:
            self.game = pyspiel.load_game(self.config.game_name, processed_game_params)
            logger.info("Game loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load game '{self.config.game_name}' with params {processed_game "numHoleCards"]:
                 try: processed_game_params[k] = int(v)
                 except (ValueError, TypeError): logger.error(f"Invalid int value for {k}: {v}"); raise
            # Оставляем строки строками
            elif k in ["betting"]:
                 processed_game_params[k] = str(v)
            # Обработка неизвестных параметров (на всякий случай)
            else:
                 logger.warning(f"Unknown game config parameter '{k}', converting to string.")
                 processed_game_params[k] = str(v)
        # -----------------------------------------------------------

        logger.info(f"Creating OpenSpiel game: {self.config.game_name} with FINAL params: {processed_game_params}")
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

    # --- Остальные методы (_define_spaces, reset, step, _process_observation и т.д.) ---
    # --- остаются БЕЗ ИЗМЕНЕНИЙ по сравнению с твоей последней версией ---
    def _reset_metrics(self):
        self.episode_rewards = {p: 0.0 for p in range(self.config.num_players)}
        self.current_hand_info = {'actions_by_player': {p: [] for p in range(self.config.num_players)}, 'involved_players': set(), 'pot_size': 0, 'current_player': -1, 'last_raiser': -1, 'can_check': {}}

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
                 self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=self.dtype)
            else: raise ValueError("Could not determine initial player.")
        except Exception as e:
             logger.error(f"Failed to determine observation size: {e}. Using fallback 512.", exc_info=True)
             self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(512,), dtype=self.dtype)

    def _get_one_observation(self, time_step, player_id):
         if not hasattr(self, 'observation_space'):
              logger.error("Observation space not defined yet in _get_one_observation!")
              return np.zeros(512, dtype=self.dtype)
         obs_shape = self.observation_space.shape
         if player_id < 0 or player_id >= len(time_step.observations.get("info_state", [])):
             logger.warning(f"Invalid player_id {player_id}. Returning zeros.")
             return np.zeros(obs_shape, dtype=self.dtype)
         try:
            base_obs_list = time_step.observations["info_state"][player_id]
            base_obs = np.array(base_obs_list, dtype=self.dtype)
            position_feature = np.array([float(player_id) / max(1, self.config.num_players - 1)], dtype=self.dtype)
            opponent_features_list = []
            for pid in range(self.config.num_players):
                if pid != player_id: opponent_features_list.extend(self.stats.get_features(pid))
            opponent_features = np.array(opponent_features_list, dtype=self.dtype)
            pot_odds_feature = np.array([self._calculate_pot_odds(time_step, player_id)], dtype=self.dtype)
            stack_to_pot_feature = np.array([self._calculate_stack_to_pot(time_step, player_id)], dtype=self.dtype)

            expected_base_len = obs_shape[0] - 1 - 1 - 1 - len(opponent_features)
            if len(base_obs) != expected_base_len:
                 logger.warning(f"Base obs length mismatch! Got {len(base_obs)}, expected {expected_base_len}. Padding/truncating.")
                 padded_base_obs = np.zeros(expected_base_len, dtype=self.dtype)
                 size = min(len(base_obs), expected_base_len)
                 padded_base_obs[:size] = base_obs[:size]
                 base_obs = padded_base_obs

            full_obs = np.concatenate([
                base_obs, position_feature, pot_odds_feature,
                stack_to_pot_feature, opponent_features
            ]).astype(self.dtype)

            if len(full_obs) != obs_shape[0]:
                 logger.error(f"FINAL obs length mismatch! Got {len(full_obs)}, expected {obs_shape[0]}. Returning zeros.")
                 return np.zeros(obs_shape, dtype=self.dtype)
            return full_obs
         except Exception as e:
            logger.error(f"Error in _get_one_params}: {e}", exc_info=True)
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
        self._define_spaces() # Определяем spaces после создания _base_env

        self._position_names = ['SB', 'BB', 'UTG', 'MP1', 'MP2', 'MP3', 'CO', 'BTN'][:self.config.num_players]
        self.current_street = 0
        self._current_time_step = None

        logger.info(f"PokerEnv initialized for {self.config.num_players} players.")

    # --- Остальные методы (_define_spaces, reset, step, etc.) ---
    # --- остаются БЕЗ ИЗМЕНЕНИЙ по сравнению с предыдущей версией ---
    # --- Убедись, что везде возвращается np.float32 ---
    def _reset_metrics(self): self.episode_rewards = {p: 0.0 for p in range(self.config.num_players)}; self.current_hand_info = {'actions_by_player': {p: [] for p in range(self.config.num_players)}, 'involved_players': set(), 'pot_size': 0, 'current_player': -1, 'last_raiser': -1, 'can_check': {}}
    def _define_spaces(self):
        num_actions = self._action_spec["num_actions"]; self.action_space = gym.spaces.Discrete(num_actions); logger.info(f"Action space: Discrete({self.action_space.n})")
        try:
            ts = self._base_env.reset(); pid = ts.observations["current_player"]; pid = 0 if pid<0 and len(ts.observations["info_state"])>0 else pid
            if pid >= 0: obs_v = self._get_one_observation(ts, pid); obs_size = len(obs_v); logger.info(f"Observation size: {obs_size}"); self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=self.dtype)
            else: raise ValueError("No initial player")
        except Exception as e: logger.error(f"Define spaces error: {e}. Fallback 512.", exc_info=True); self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(512,), dtype=self.dtype)
    def _get_one_observation(self, ts, pid):
         shape = self.observation_space.shape if hasattr(self,'observation_space') else (512,);
         if pid < 0 or pid >= len(ts.observations.get("info_state",[])): return np.zeros(shape, dtype=self.dtype)
         try:
            base_obs = np.array(ts.observations["info_state"][pid], dtype=self.dtype)
            pos_f = np._observation for player {player_id}: {e}", exc_info=True)
            return np.zeros(obs_shape, dtype=self.dtype)

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
            return obs_vector.astype(self.dtype), {}
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
             return obs.astype(self.dtype), 0.0, True, False, info

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
            if new_street != self.current_street: self.current_street = new_street

            reward = float(self._current_time_step.rewards[current_player])
            terminated = self._current_time_step.lastarray([float(pid)/max(1,self.config.num_players-1)], dtype=self.dtype)
            opp_f_list = [f for i in range(self.config.num_players) if i!=pid for f in self.stats.get_features(i)]; opp_f = np.array(opp_f_list, dtype=self.dtype)
            po_f = np.array([self._calculate_pot_odds(ts, pid)], dtype=self.dtype); spr_f = np.array([self._calculate_stack_to_pot(ts, pid)], dtype=self.dtype)
            exp_base_len = shape[0]-len(pos_f)-len(po_f)-len(spr_f)-len(opp_f)
            if len(base_obs) != exp_base_len: base_obs = np.pad(base_obs, (0, exp_base_len-len(base_obs)), mode='constant').astype(self.dtype)[:exp_base_len] # Pad/truncate base
            full_obs = np.concatenate([base_obs, pos_f, po_f, spr_f, opp_f]).astype(self.dtype)
            return full_obs if len(full_obs)==shape[0] else np.zeros(shape, dtype=self.dtype) # Final check
         except Exception as e: logger.error(f"Get obs error p{pid}: {e}", exc_info=True); return np.zeros(shape, dtype=self.dtype)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); logger.info("Resetting env...");
        try:
            self._current_time_step = self._base_env.reset(); self.current_street = 0; self._reset_metrics()
            for p()
            truncated = False
            obs_vector = self._get_one_observation(self._current_time_step, next_player)

            if terminated:
                self.stats.finalize_hand_stats(self.current_hand_info['involved_players'])
                for p, r in enumerate(self._current_time_step.rewards):
                     self.episode_rewards[p] = self.episode_rewards.get(p, 0.0) + float(r)

            info = self._enhance_info({}, next_player)
            return obs_vector.astype(self.dtype), reward, terminated, truncated, info

        except Exception as e:
            logger.error(f"Error during step: {e}", exc_info=True)
            return np.zeros(self.observation_space.shape, dtype=self.dtype), 0.0, True, False, {"error": str(e)}

    def _get_state(self): return getattr(self._base_env, '_state', None)
    def _get_pot_size(self, time_step): state = self._get_state(); return float(state.pot()) if state and hasattr(state, 'pot') else 0.0
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
    def _get_street(self, time_step): state = self._get_state(); return int(state.round()) if state in range(self.config.num_players): self.stats.record_opportunity(p, p!=1, True)
            pid = self._current_time_step.observations["current_player"]; pid = 0 if pid<0 else pid
            obs = self._get_one_observation(self._current_time_step, pid)
            return obs, {}
        except Exception as e: logger.error(f"Reset error: {e}", exc_info=True); return np.zeros(self.observation_space.shape, dtype=self.dtype), {}
    def step(self, action):
        if isinstance(action, np.generic): action = int(action)
        p_before = self._current_time_step.observations["current_player"] if self._current_time_step else -1; logger.debug(f"P{p_before} action: {action}")
        if self._current_time_step is None or self._current_time_step. and hasattr(state, 'round') else 0
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
             except:last(): obs, info = self.reset(); return obs, 0.0, True, False, info
        curr_p = self._current_time_step.observations["current_player"]; legal = self._current_time_step.observations["legal_actions"][curr_p]
        if action not in legal: action = self.np_random.choice(legal) if legal else 0; logger.warning(f"Illegal action corrected to {action}")
        try:
            act_type = self._get_action_type(action); is_vol = (self.current_street==0 and act_type!='fold') or (self.current_street>0 and pass
         return 100.0
    def _enhance_info(self, info, next_player_id):
        if hasattr(self,'_current_time_step') and self._current_time_step and not self._current_time_step.last() and next_player_id >= 0:
             try: info['legal_actions'] = self._current_time_step.observations["legal_actions"][next_player_id]
             except: info['legal_actions'] = []
        # Возвращаем статы как list для act_type!='fold')
            self.stats.update_on_action(curr_p, act_type, self.current_street, is_vol); self.current_hand_info['actions_by_player'][curr_p].append(action); self.current_hand_info['involved_players'].add(curr_p)
            self._current_time_step = self._base_env.step([action])
            self. JSON-сериализации
        info['metrics'] = {p: self.stats.get_features(p).tolist() for p in range(self.config.num_players)}
        return info
    def render(self, mode='human'): state = self._get_state(); rendered = str(state) if state else "current_hand_info['pot_size'] = self._get_pot_size(self._current_time_step); next_p = self._current_time_step.observations["current_player"]; self.current_hand_info['current_player'] = next_p
            new_s = self._get_street(self._current_time_step);
            if new_s != self.current_street: self.current_No state."; print(rendered); return rendered
    def close(self): logger.info("Closing PokerEnv"); self._base_env = None; self.game = None

# environment_fixed.py
"""
Fixed version of the Poker environment for RL training.
This version includes improved error handling, memory optimizations,
and compatibility fixes for OpenSpiel and Gymnasium.
"""

import numpy as np
import gymnasium as gym
from gymnasium.utils import seeding
from gymnasium.spaces import Box, Discrete, Dict as GymDict
from open_spiel.python import rl_environment
import pyspiel
from collections import deque
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
import traceback
import gc
import time

# Ensure consistent logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class OpponentStats:
    """Class for tracking and updating basic opponent statistics."""
    def __init__(self, num_players, dtype=np.float32):
        """Initialize opponent statistics tracker."""
        self.num_players = num_players
        self.stats = {}
        self.feature_size = 4  # VPIP, PFR, AF, Hands Played
        self.dtype = dtype
        self.reset()

    def reset(self):
        """Reset all opponent statistics."""
        self.stats = {}
        for i in range(self.num_players):
            self._ensure_player_stats(i)

    def _ensure_player_stats(self, player_id):
        """Ensure a player has stats initialized."""
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
        """Return normalized statistics for a player as numpy array."""
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

        # Safe calculation with fallbacks
        vpip = float(vpip_actions / vpip_opps) if vpip_opps > 0 else 0.0
        pfr = float(pfr_actions / pfr_opps) if pfr_opps > 0 else 0.0
        af = float(agg_actions / pass_actions) if pass_actions > 0 else (5.0 if agg_actions > 0 else 1.0)

        # Normalize all values to [0,1] range
        norm_vpip = np.clip(vpip, 0.0, 1.0)
        norm_pfr = np.clip(pfr, 0.0, 1.0)
        norm_af = np.clip(af / 5.0, 0.0, 1.0)  # Scale AF to ~[0,1]
        norm_hands = np.clip(hands / 100.0, 0.0, 1.0)  # Normalize hands count

        return np.array([norm_vpip, norm_pfr, norm_af, norm_hands], dtype=self.dtype)

    def record_opportunity(self, player_id, is_vpip_opp, is_pfr_opp):
        """Record an opportunity for VPIP/PFR."""
        player_id_int = int(player_id)
        self._ensure_player_stats(player_id_int)
        if is_vpip_opp: 
            self.stats[player_id_int]['vpip_opportunities'] += 1
        if is_pfr_opp: 
            self.stats[player_id_int]['pfr_opportunities'] += 1

    def update_on_action(self, player_id, action_type, street, is_voluntary_action):
        """Update action counters based on player actions."""
        player_id_int = int(player_id)
        self._ensure_player_stats(player_id_int)
        stats = self.stats[player_id_int]
        
        # VPIP: Count first voluntary action (not fold) on preflop
        if street == 0 and is_voluntary_action and not stats['vpip_acted_this_hand']:
            stats['vpip_actions'] += 1
            stats['vpip_acted_this_hand'] = True
            
        # PFR: Count first raise on preflop
        if street == 0 and action_type == 'raise' and not stats['pfr_acted_this_hand']:
            stats['pfr_actions'] += 1
            stats['pfr_acted_this_hand'] = True
            
        # AF: Count aggressive (bet/raise) and passive (call) actions POSTFLOP
        if street > 0:
            if action_type == 'raise': 
                stats['agg_actions'] += 1
            elif action_type == 'call': 
                stats['pass_actions'] += 1  # Only count calls, not checks

    def finalize_hand_stats(self, involved_players):
        """Update end-of-hand statistics and reset hand flags."""
        for player_id in involved_players:
            player_id_int = int(player_id)
            self._ensure_player_stats(player_id_int)
            self.stats[player_id_int]['hands'] += 1
            # Reset hand-specific flags for next hand
            self.stats[player_id_int]['vpip_acted_this_hand'] = False
            self.stats[player_id_int]['pfr_acted_this_hand'] = False


class PokerEnv(gym.Env):
    """
    Poker environment for RLlib using OpenSpiel with action masking.
    Implements Gymnasium interface with Dict observation space containing
    both observations and action masks.
    """
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}
    
    # Define action mapping for better debugging
    ACTION_NAMES = {
        0: "fold",
        1: "call/check",
        2: "raise"
    }

    def __init__(self, env_config: dict):
        """Initialize the poker environment."""
        super().__init__()
        logger.info("Initializing PokerEnv with action masking...")

        try:
            # Extract config
            if "config" not in env_config:
                if "env_config" in env_config: 
                    env_config = env_config["env_config"]
                else: 
                    raise ValueError("Missing 'config' key in env_config dictionary")
                    
            if "config" not in env_config:
                raise ValueError("Missing 'config' key after checking nested 'env_config'")
                
            self.config = env_config["config"]
            
            # Get dtype or use float32
            self.dtype = env_config.get("dtype", np.float32)
            if not isinstance(self.dtype, type) or not np.issubdtype(self.dtype, np.floating):
                logger.warning(f"Invalid dtype: {self.dtype}. Using np.float32")
                self.dtype = np.float32

            # OpenSpiel initialization
            game_params = self.config.game_config if hasattr(self.config, 'game_config') else {}
            game_name = self.config.game_name if hasattr(self.config, 'game_name') else "leduc_poker"
            
            # Process game parameters for OpenSpiel
            processed_params = self._process_game_params(game_params)
            logger.info(f"Loading OpenSpiel game: '{game_name}' with params: {processed_params}")
            
            # Create the game
            self.game = pyspiel.load_game(game_name, processed_params)
            logger.info(f"Game '{game_name}' loaded successfully (Num players: {self.game.num_players()})")
            
            # Create OpenSpiel RL Environment
            self._base_env = rl_environment.Environment(game=self.game, include_full_state=False)
            logger.info("OpenSpiel RL Environment created successfully")
            
            # Get action and observation specifications
            self._action_spec = self._base_env.action_spec()
            self._observation_spec = self._base_env.observation_spec()
            
            # Get actual number of players
            num_players = self.game.num_players()
            
            # Initialize opponent statistics
            self.stats = OpponentStats(num_players, dtype=self.dtype)
            
            # Define Gymnasium spaces
            self._define_spaces(num_players)
            
            # Helper attributes
            self._position_names = ['SB', 'BB', 'UTG', 'MP1', 'MP2', 'MP3', 'CO', 'BTN'][:num_players]
            self.current_street = 0
            self._current_time_step = None  # Stores current TimeStep from OpenSpiel
            
            # Initialize Gymnasium RNG
            self.np_random = None
            self.seed()
            
            # Add spec attribute for RLlib/Gymnasium>=0.26 compatibility
            try:
                from gymnasium.envs.registration import EnvSpec
                self.spec = EnvSpec(id="PokerEnv-v0", max_episode_steps=200)
                logger.info(f"Set EnvSpec with max_episode_steps={self.spec.max_episode_steps}")
            except ImportError:
                logger.warning("Could not import EnvSpec, using fallback")
                self.spec = type('obj', (object,), {'max_episode_steps': float('inf')})()
                
            logger.info(f"PokerEnv initialized for {num_players} players")
            logger.info(f"Observation Space: {self.observation_space}")
            logger.info(f"Action Space: {self.action_space}")
            
        except Exception as e:
            logger.error(f"Error initializing PokerEnv: {e}")
            logger.error(traceback.format_exc())
            raise

    def _process_game_params(self, game_params):
        """Process game parameters for OpenSpiel format."""
        processed_params = {}
        
        for k, v in game_params.items():
            if k in ["numBoardCards", "blind", "stack"] and isinstance(v, (list, tuple, np.ndarray)):
                # Convert lists to space-separated strings for OpenSpiel
                processed_params[k] = " ".join(map(str, v))
            elif isinstance(v, bool):
                # Convert booleans to lowercase strings
                processed_params[k] = str(v).lower()
            elif isinstance(v, (int, float, np.number)):
                # Keep numeric values as is
                processed_params[k] = v
            else:
                # Convert everything else to strings
                processed_params[k] = str(v)
                
        return processed_params

    def seed(self, seed=None):
        """Set the seed for the environment's RNG."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset_metrics(self):
        """Reset metrics for a new episode."""
        self.episode_rewards = {p: 0.0 for p in range(self.game.num_players())}
        self.current_hand_info = {
            'actions_by_player': {p: [] for p in range(self.game.num_players())},
            'involved_players': set(),
            'pot_size': 0,
            'current_player': -1,
            'last_raiser': -1,
            'can_check': {}
        }

    def _define_spaces(self, num_players):
        """Define action_space and observation_space for Gymnasium."""
        try:
            # 1. Action space
            num_actions = self._action_spec["num_actions"]
            self.action_space = Discrete(num_actions)
            
            # 2. Observation space (Dict)
            # Get example observation to determine sizes
            temp_ts = self._base_env.reset()
            player_id = temp_ts.observations["current_player"]
            
            # Handle chance nodes until we get a player observation
            timeout = 10  # Safety to prevent infinite loops
            attempts = 0
            while player_id == pyspiel.PlayerId.CHANCE and attempts < timeout:
                temp_ts = self._base_env.step([])
                player_id = temp_ts.observations["current_player"]
                attempts += 1
                
            if player_id < 0 or player_id == pyspiel.PlayerId.TERMINAL:
                logger.warning("Could not get valid player observation, using fallback sizes")
                base_obs_size = 800  # Large fallback size for safety
                legal_actions_size = num_actions
            else:
                # Get sizes from actual observation
                base_obs_size = len(temp_ts.observations["info_state"][player_id])
                legal_actions_size = num_actions
                
            # Add opponent stats features to observation
            stats_size = self.stats.feature_size * (num_players - 1)
            total_obs_size = base_obs_size + stats_size
            
            # Create Box spaces with appropriate dtypes
            self.observation_space = GymDict({
                "obs": Box(
                    low=-float('inf'),
                    high=float('inf'),
                    shape=(total_obs_size,),
                    dtype=self.dtype
                ),
                "action_mask": Box(
                    low=0,
                    high=1,
                    shape=(legal_actions_size,),
                    dtype=np.int8  # Use int8 for masks to save memory
                )
            })
            
            # Store sizes for reference
            self._base_obs_size = base_obs_size
            self._stats_size = stats_size
            self._total_obs_size = total_obs_size
            
            logger.info(f"Observation space configured - base: {base_obs_size}, stats: {stats_size}, total: {total_obs_size}")
            
        except Exception as e:
            logger.error(f"Error defining spaces: {e}")
            logger.error(traceback.format_exc())
            raise

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Optional seed for the RNG
            options: Optional configuration parameters
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        try:
            if seed is not None:
                self.seed(seed)
                
            # Reset OpenSpiel environment
            self._current_time_step = self._base_env.reset()
            self._reset_metrics()
            self.current_street = 0
            
            # Handle chance nodes until we get a player observation
            while self._current_player() == pyspiel.PlayerId.CHANCE:
                self._current_time_step = self._base_env.step([])
                
            # If the game ended immediately for some reason
            if self.is_terminal():
                logger.warning("Game terminated immediately after reset")
                return self._get_observation(), {}
                
            # Return observation and info
            return self._get_observation(), {}
            
        except Exception as e:
            logger.error(f"Error in reset: {e}")
            logger.error(traceback.format_exc())
            # Try to recover with a clean reset
            try:
                self._base_env = rl_environment.Environment(game=self.game, include_full_state=False)
                self._current_time_step = self._base_env.reset()
                self._reset_metrics()
                return self._get_observation(), {"error": str(e)}
            except Exception as recovery_error:
                logger.error(f"Failed to recover from reset error: {recovery_error}")
                # Return empty observation as last resort
                empty_obs = {
                    "obs": np.zeros(self._total_obs_size, dtype=self.dtype),
                    "action_mask": np.ones(self.action_space.n, dtype=np.int8)
                }
                return empty_obs, {"critical_error": str(e)}

    def _current_player(self):
        """Get the current player ID."""
        if self._current_time_step is None:
            return pyspiel.PlayerId.TERMINAL
            
        return self._current_time_step.observations["current_player"]

    def is_terminal(self):
        """Check if the current state is terminal."""
        current_player = self._current_player()
        return current_player == pyspiel.PlayerId.TERMINAL

    def _get_legal_actions_mask(self):
        """
        Get a mask for legal actions.
        1 = legal action, 0 = illegal action
        """
        if self.is_terminal():
            # No legal actions in terminal states
            return np.zeros(self.action_space.n, dtype=np.int8)
            
        current_player = self._current_player()
        if current_player < 0:
            # No legal actions for chance/invalid players
            return np.zeros(self.action_space.n, dtype=np.int8)
            
        legal_actions = self._current_time_step.observations["legal_actions"][current_player]
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        mask[legal_actions] = 1
        return mask

    def _get_observation(self):
        """
        Get observation dictionary with 'obs' and 'action_mask'.
        """
        if self._current_time_step is None or self.is_terminal():
            # Return zeros if terminal or not initialized
            return {
                "obs": np.zeros(self._total_obs_size, dtype=self.dtype),
                "action_mask": np.zeros(self.action_space.n, dtype=np.int8)
            }
            
        current_player = self._current_player()
        if current_player < 0:
            # Invalid observation for chance/terminal
            return {
                "obs": np.zeros(self._total_obs_size, dtype=self.dtype),
                "action_mask": np.zeros(self.action_space.n, dtype=np.int8)
            }
            
        # Get base observation from OpenSpiel
        base_obs = np.array(self._current_time_step.observations["info_state"][current_player], 
                           dtype=self.dtype)
        
        # Pad if necessary (in case the observation is smaller than expected)
        if len(base_obs) < self._base_obs_size:
            logger.warning(f"Base observation size mismatch: {len(base_obs)} < {self._base_obs_size}")
            pad_size = self._base_obs_size - len(base_obs)
            base_obs = np.pad(base_obs, (0, pad_size), 'constant')
        elif len(base_obs) > self._base_obs_size:
            logger.warning(f"Base observation size mismatch: {len(base_obs)} > {self._base_obs_size}")
            base_obs = base_obs[:self._base_obs_size]
            
        # Get opponent stats features
        opponent_stats = []
        num_players = self.game.num_players()
        
        for player_id in range(num_players):
            if player_id != current_player:
                opponent_stats.append(self.stats.get_features(player_id))
                
        # Concatenate opponent stats
        opponent_stats_flat = np.concatenate(opponent_stats).astype(self.dtype)
        
        # Ensure opponent stats size matches expected size
        if len(opponent_stats_flat) != self._stats_size:
            logger.warning(f"Stats size mismatch: {len(opponent_stats_flat)} != {self._stats_size}")
            # Fix size if needed
            if len(opponent_stats_flat) < self._stats_size:
                opponent_stats_flat = np.pad(opponent_stats_flat, (0, self._stats_size - len(opponent_stats_flat)), 'constant')
            else:
                opponent_stats_flat = opponent_stats_flat[:self._stats_size]
                
        # Combine base observation and opponent stats
        full_obs = np.concatenate([base_obs, opponent_stats_flat])
        
        # Get action mask
        action_mask = self._get_legal_actions_mask()
        
        return {
            "obs": full_obs,
            "action_mask": action_mask
        }

    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            observation: New observation
            reward: Reward received
            terminated: Whether the episode is terminated
            truncated: Whether the episode was truncated (not used)
            info: Additional information
        """
        try:
            if self.is_terminal():
                logger.warning("Step called in terminal state")
                return (
                    self._get_observation(),
                    0.0,
                    True,
                    False,
                    {"error": "Step called in terminal state"}
                )
                
            current_player = self._current_player()
            if current_player < 0:
                logger.warning(f"Step called with invalid player: {current_player}")
                return (
                    self._get_observation(),
                    0.0,
                    False,
                    False,
                    {"error": f"Invalid player: {current_player}"}
                )
                
            # Update hand information for statistics
            self._update_hand_info_pre_action(current_player, action)
            
            # Take the action
            self._current_time_step = self._base_env.step([action])
            
            # Update statistics
            self._update_statistics(current_player, action)
            
            # Handle chance nodes
            while self._current_player() == pyspiel.PlayerId.CHANCE:
                self._current_time_step = self._base_env.step([])
                
            # Calculate rewards if terminal
            rewards = {p: 0.0 for p in range(self.game.num_players())}
            terminated = self.is_terminal()
            
            if terminated:
                returns = self._current_time_step.rewards
                for p in range(self.game.num_players()):
                    rewards[p] = returns[p]
                    self.episode_rewards[p] += returns[p]
                # Finalize hand statistics
                self.stats.finalize_hand_stats(self.current_hand_info['involved_players'])
                
            # Return observation and results
            return (
                self._get_observation(),
                rewards[current_player],
                terminated,
                False,
                self._get_info()
            )
            
        except Exception as e:
            logger.error(f"Error in step: {e}")
            logger.error(traceback.format_exc())
            # Try to recover
            return (
                self._get_observation(),
                0.0,
                True,  # Force termination on error
                False,
                {"critical_error": str(e)}
            )

    def _update_hand_info_pre_action(self, player_id, action):
        """Update hand information before taking an action."""
        # Record player involvement
        self.current_hand_info['involved_players'].add(player_id)
        self.current_hand_info['current_player'] = player_id
        
        # Store action for this player
        legal_actions = self._current_time_step.observations["legal_actions"][player_id]
        action_str = self.ACTION_NAMES.get(action, str(action))
        
        self.current_hand_info['actions_by_player'][player_id].append({
            'action': action,
            'action_name': action_str,
            'street': self.current_street,
            'legal_actions': legal_actions
        })

    def _update_statistics(self, player_id, action):
        """Update player statistics based on action taken."""
        try:
            # Basic action categorization
            is_fold = action == 0
            is_call = action == 1
            is_raise = action == 2
            is_voluntary_action = is_call or is_raise
            
            # Determine action type for statistics
            if is_fold:
                action_type = 'fold'
            elif is_call:
                action_type = 'call'
            elif is_raise:
                action_type = 'raise'
            else:
                action_type = 'unknown'
                
            # Update statistics
            self.stats.update_on_action(
                player_id=player_id,
                action_type=action_type,
                street=self.current_street,
                is_voluntary_action=is_voluntary_action
            )
            
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")

    def _get_info(self):
        """Get additional information about the environment state."""
        info = {
            "episode_rewards": self.episode_rewards.copy(),
            "current_street": self.current_street,
            "pot_size": self.current_hand_info['pot_size'],
        }
        
        # Add player positions if in active game state
        current_player = self._current_player()
        if current_player >= 0 and current_player < self.game.num_players():
            player_position = self._get_position_name(current_player)
            info["current_player_position"] = player_position
            
        # Add terminal state information if applicable
        if self.is_terminal():
            info["terminal"] = True
            info["final_rewards"] = self._current_time_step.rewards
            
        return info

    def _get_position_name(self, player_id):
        """Get the position name for a player ID."""
        if 0 <= player_id < len(self._position_names):
            return self._position_names[player_id]
        return f"Player{player_id}"

    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode ('human' or 'ansi')
            
        Returns:
            Rendered string or None
        """
        if self._current_time_step is None:
            return "Environment not initialized"
            
        if mode == 'ansi':
            # Simple text representation
            lines = []
            lines.append("=" * 50)
            
            if self.is_terminal():
                lines.append("GAME OVER")
                lines.append(f"Rewards: {self._current_time_step.rewards}")
            else:
                current_player = self._current_player()
                if current_player >= 0:
                    position = self._get_position_name(current_player)
                    lines.append(f"Current Player: {current_player} ({position})")
                    
                    legal_actions = self._current_time_step.observations["legal_actions"][current_player]
                    legal_action_names = [self.ACTION_NAMES.get(a, str(a)) for a in legal_actions]
                    lines.append(f"Legal Actions: {legal_action_names}")
                else:
                    lines.append(f"Current Player: {current_player} (Chance/Terminal)")
                    
            # Add history
            lines.append("\nAction History:")
            for player, actions in self.current_hand_info['actions_by_player'].items():
                if actions:
                    position = self._get_position_name(player)
                    lines.append(f"Player {player} ({position}):")
                    for act in actions:
                        lines.append(f"  Street {act['street']}: {act['action_name']}")
                        
            lines.append("=" * 50)
            rendered = "\n".join(lines)
            
            if mode == 'human':
                print(rendered)
                return None
            else:
                return rendered
        else:
            return "Rendering mode not supported"

    def close(self):
        """Clean up environment resources."""
        # Clear large objects
        self._current_time_step = None
        # Force garbage collection
        gc.collect()
        super().close()

def create_poker_env(env_config):
    """Factory function to create a poker environment."""
    return PokerEnv(env_config)

#!/usr/bin/env python3
"""
Test script for the Poker RL environment.
Run this script to verify the environment works correctly
before starting a full training run.
"""

import argparse
import logging
import numpy as np
import time
import traceback
import json
from pathlib import Path

from open_spiel.python import rl_environment
import pyspiel

from config import PokerConfig
from environment_fixed import PokerEnv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('env_test.log')
    ]
)
logger = logging.getLogger("env_test")

def test_openspiel_directly():
    """Test OpenSpiel's universal_poker game directly."""
    logger.info("Testing OpenSpiel universal_poker game directly...")
    
    try:
        # Create a simple game configuration
        game_string = (
            "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,"
            "blind=50 100,stack=10000 10000,"
            "numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1)"
        )
        
        # Load the game
        game = pyspiel.load_game(game_string)
        logger.info(f"Game created successfully: {game}")
        
        # Create a new initial state
        state = game.new_initial_state()
        logger.info(f"Initial state: {state}")
        
        # Process chance events until we're at a decision point
        while state.is_chance_node():
            legal_actions = state.legal_actions()
            action = np.random.choice(legal_actions)
            state.apply_action(action)
            logger.info(f"Applied chance action: {action}")
        
        # Print current state
        logger.info(f"Current player: {state.current_player()}")
        logger.info(f"Legal actions: {state.legal_actions()}")
        
        # Test success
        return True
    except Exception as e:
        logger.error(f"OpenSpiel test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_rl_environment():
    """Test OpenSpiel's RL environment wrapper."""
    logger.info("Testing OpenSpiel RL environment...")
    
    try:
        # Create a simple game configuration
        game_string = (
            "universal_poker(betting=nolimit,numPlayers=2,numRounds=4,"
            "blind=50 100,stack=10000 10000,"
            "numSuits=4,numRanks=13,numHoleCards=2,numBoardCards=0 3 1 1)"
        )
        
        # Create the game
        game = pyspiel.load_game(game_string)
        
        # Create RL environment
        env = rl_environment.Environment(game, include_full_state=True)
        logger.info("RL environment created successfully")
        
        # Reset the environment
        time_step = env.reset()
        logger.info(f"Reset successful, current player: {time_step.observations['current_player']}")
        
        # Process if the initial state is a chance node
        while time_step.observations["current_player"] == pyspiel.PlayerId.CHANCE:
            time_step = env.step([])
            logger.info("Processed chance node")
        
        # Get current player
        current_player = time_step.observations["current_player"]
        logger.info(f"Current player after chance: {current_player}")
        
        # Get legal actions
        if current_player >= 0:
            legal_actions = time_step.observations["legal_actions"][current_player]
            logger.info(f"Legal actions: {legal_actions}")
            
            # Take a random action
            if legal_actions:
                action = np.random.choice(legal_actions)
                next_time_step = env.step([action])
                logger.info(f"Took action {action}, rewards: {next_time_step.rewards}")
        
        # Test success
        return True
    except Exception as e:
        logger.error(f"RL environment test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def run_single_episode(env, max_steps=200, render=False):
    """Run a single episode in the environment."""
    # Reset the environment
    start_time = time.time()
    obs = env.reset()
    
    # Track episode information
    episode_reward = 0
    step_count = 0
    terminated = False
    truncated = False
    
    # Store all observations for analysis
    observations = []
    actions_taken = []
    
    # Run until episode ends or max steps reached
    while not terminated and not truncated and step_count < max_steps:
        # Get valid actions from mask
        if isinstance(obs, dict) and "action_mask" in obs:
            action_mask = obs["action_mask"]
            valid_actions = np.where(action_mask > 0)[0]
        else:
            # Fallback if no mask
            valid_actions = np.arange(env.action_space.n)
        
        if len(valid_actions) == 0:
            logger.warning(f"No valid actions at step {step_count}")
            valid_actions = [0]  # Default to fold if no valid actions
        
        # Choose a random valid action
        action = np.random.choice(valid_actions)
        
        # Take the action
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Optional rendering
        if render and step_count % 5 == 0:
            env.render(mode='ansi')
        
        # Store data
        observations.append(obs)
        actions_taken.append(action)
        episode_reward += reward
        step_count += 1
        
        # Update observation
        obs = next_obs
    
    # Calculate timing and stats
    duration = time.time() - start_time
    steps_per_second = step_count / duration if duration > 0 else 0
    
    episode_info = {
        "reward": float(episode_reward),
        "steps": step_count,
        "duration_seconds": duration,
        "steps_per_second": steps_per_second,
        "terminated": terminated,
        "truncated": truncated
    }
    
    return episode_info, observations, actions_taken

def test_custom_environment(num_episodes=10, render=False):
    """Test our custom PokerEnv environment."""
    logger.info("Testing custom PokerEnv environment...")
    
    try:
        # Create configuration
        config = PokerConfig()
        
        # Try different player counts
        player_counts = [2, 4, 8]
        results = {}
        
        for num_players in player_counts:
            logger.info(f"Testing with {num_players} players...")
            config.num_players = num_players
            
            # Create environment
            env_config = {"config": config}
            env = PokerEnv(env_config)
            
            # Log spaces
            logger.info(f"Observation space: {env.observation_space}")
            logger.info(f"Action space: {env.action_space}")
            
            # Run episodes
            episode_results = []
            for ep in range(num_episodes):
                logger.info(f"Running episode {ep+1}/{num_episodes} with {num_players} players")
                episode_info, _, _ = run_single_episode(env, render=render)
                logger.info(f"Episode {ep+1} result: reward={episode_info['reward']}, steps={episode_info['steps']}")
                episode_results.append(episode_info)
            
            # Store results
            results[f"{num_players}_players"] = {
                "episodes": episode_results,
                "mean_reward": np.mean([ep["reward"] for ep in episode_results]),
                "mean_steps": np.mean([ep["steps"] for ep in episode_results]),
                "mean_duration": np.mean([ep["duration_seconds"] for ep in episode_results])
            }
            
            # Close environment
            env.close()
        
        # Save results
        results_dir = Path("test_results")
        results_dir.mkdir(exist_ok=True)
        results_path = results_dir / f"env_test_{int(time.time())}.json"
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Test results saved to {results_path}")
        
        # Log summary
        logger.info("Environment test summary:")
        for player_count, res in results.items():
            logger.info(f"{player_count}: {res['mean_steps']:.1f} steps, {res['mean_reward']:.2f} reward")
        
        return True
    except Exception as e:
        logger.error(f"Custom environment test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_observation_memory():
    """Test observation memory usage to check for leaks."""
    logger.info("Testing observation memory usage...")
    
    try:
        import psutil
        import gc
        
        process = psutil.Process()
        
        # Create environment
        config = PokerConfig()
        config.num_players = 8  # Larger game to stress test
        
        env_config = {"config": config}
        env = PokerEnv(env_config)
        
        # Initial memory usage
        gc.collect()  # Force garbage collection
        initial_memory = process.memory_info().rss / (1024 * 1024)
        logger.info(f"Initial memory usage: {initial_memory:.2f} MB")
        
        # Run many resets and steps
        num_cycles = 100
        for i in range(num_cycles):
            obs = env.reset()
            
            # Take 10 steps in each episode
            for _ in range(10):
                if isinstance(obs, dict) and "action_mask" in obs:
                    action_mask = obs["action_mask"]
                    valid_actions = np.where(action_mask > 0)[0]
                    if len(valid_actions) == 0:
                        break
                    action = np.random.choice(valid_actions)
                else:
                    action = env.action_space.sample()
                
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            
            # Check memory every 10 cycles
            if (i + 1) % 10 == 0:
                gc.collect()  # Force garbage collection
                current_memory = process.memory_info().rss / (1024 * 1024)
                logger.info(f"After {i+1} cycles: {current_memory:.2f} MB "
                           f"(change: {current_memory - initial_memory:.2f} MB)")
        
        # Final memory check
        gc.collect()
        final_memory = process.memory_info().rss / (1024 * 1024)
        memory_increase = final_memory - initial_memory
        
        logger.info(f"Final memory usage: {final_memory:.2f} MB")
        logger.info(f"Memory increase: {memory_increase:.2f} MB")
        
        # A large increase might indicate a memory leak
        if memory_increase > 100:
            logger.warning(f"Possible memory leak: {memory_increase:.2f} MB increase after {num_cycles} cycles")
        else:
            logger.info(f"Memory usage appears stable (increase: {memory_increase:.2f} MB)")
        
        # Clean up
        env.close()
        
        return memory_increase < 100  # Return success if increase is reasonable
    except Exception as e:
        logger.error(f"Memory test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def test_multi_environment(num_envs=4):
    """Test running multiple environments in parallel."""
    logger.info(f"Testing {num_envs} environments in parallel...")
    
    try:
        # Create environments
        envs = []
        for i in range(num_envs):
            config = PokerConfig()
            config.num_players = 2  # Use small game for speed
            env_config = {"config": config}
            env = PokerEnv(env_config)
            envs.append(env)
        
        # Initial resets
        observations = [env.reset() for env in envs]
        
        # Run 20 steps in parallel
        for step in range(20):
            actions = []
            
            # Determine actions for all environments
            for i, obs in enumerate(observations):
                if isinstance(obs, dict) and "action_mask" in obs:
                    action_mask = obs["action_mask"]
                    valid_actions = np.where(action_mask > 0)[0]
                    if len(valid_actions) == 0:
                        actions.append(0)  # Default to fold
                    else:
                        actions.append(np.random.choice(valid_actions))
                else:
                    actions.append(envs[i].action_space.sample())
            
            # Take steps in all environments
            new_observations = []
            rewards = []
            dones = []
            
            for i, (env, action) in enumerate(zip(envs, actions)):
                obs, reward, terminated, truncated, _ = env.step(action)
                new_observations.append(obs)
                rewards.append(reward)
                dones.append(terminated or truncated)
            
            # Update observations
            observations = new_observations
            
            # Reset finished environments
            for i, done in enumerate(dones):
                if done:
                    observations[i] = envs[i].reset()
            
            logger.info(f"Step {step+1}: {sum(dones)} environments reset")
        
        # Clean up
        for env in envs:
            env.close()
        
        logger.info("Multi-environment test completed successfully")
        return True
    except Exception as e:
        logger.error(f"Multi-environment test failed: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description="Test the Poker RL environment")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to test")
    parser.add_argument("--render", action="store_true", help="Render environment during testing")
    parser.add_argument("--memory", action="store_true", help="Run memory usage test")
    parser.add_argument("--multi", action="store_true", help="Test multiple environments")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Set log level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=== Poker Environment Testing Suite ===")
    
    # Track test results
    test_results = {}
    
    # Determine which tests to run
    run_all = args.all or not any([args.memory, args.multi])
    
    # OpenSpiel direct test
    logger.info("\n=== Testing OpenSpiel Directly ===")
    test_results["openspiel_direct"] = test_openspiel_directly()
    
    # OpenSpiel RL Environment test
    logger.info("\n=== Testing OpenSpiel RL Environment ===")
    test_results["openspiel_rl_env"] = test_rl_environment()
    
    # Custom environment test
    logger.info("\n=== Testing Custom Poker Environment ===")
    test_results["custom_env"] = test_custom_environment(
        num_episodes=args.episodes, 
        render=args.render
    )
    
    # Memory test
    if run_all or args.memory:
        logger.info("\n=== Testing Memory Usage ===")
        test_results["memory_test"] = test_observation_memory()
    
    # Multi-environment test
    if run_all or args.multi:
        logger.info("\n=== Testing Multiple Environments ===")
        test_results["multi_env"] = test_multi_environment()
    
    # Print summary
    logger.info("\n=== Test Summary ===")
    all_passed = True
    for test_name, result in test_results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")
        all_passed = all_passed and result
    
    # Final verdict
    if all_passed:
        logger.info("\n✅ All tests passed! Environment appears ready for training.")
        exit_code = 0
    else:
        logger.error("\n❌ Some tests failed. Review the logs and fix issues before training.")
        exit_code = 1
    
    return exit_code

if __name__ == "__main__":
    exit(main())

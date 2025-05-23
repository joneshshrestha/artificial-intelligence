import gymnasium as gym
import time

# Test visual rendering with gymnasium's built-in FrozenLake
print("Testing visual simulation with gymnasium's FrozenLake...")

try:
    # Create environment with visual rendering
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=True, render_mode="human")
    
    # Reset and test one step
    state, _ = env.reset()
    print(f"Initial state: {state}")
    
    # Take one action and see if visual works
    action = env.action_space.sample()  # Random action
    state, reward, done, truncated, info = env.step(action)
    print(f"After action {action}: state={state}, reward={reward}")
    
    # Keep window open for a few seconds
    time.sleep(3)
    env.close()
    
    print("Visual simulation test SUCCESSFUL!")
    
except Exception as e:
    print(f"Visual simulation test FAILED: {e}") 
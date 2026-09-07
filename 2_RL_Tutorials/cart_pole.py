import gymnasium as gym

# Print the environments we can create
gym.pprint_registry()

# Create a simple environment perfect for beginners
env = gym.make('CartPole-v1', render_mode="human")

# Reset environment
observation, info = env.reset()
print(f"Starting observation: {observation}")

episode_over = False
total_reward = 0

while not episode_over:
    # Choose an action
    action = env.action_space.sample()  # Random Action choice from the action space

    # Take the action and see what happens
    observation, reward, terminated, truncated, info = env.step(action)

    # Increment reward
    total_reward += reward

    # Check for fail conditions
    episode_over = terminated or truncated

# Close environment
print(f"Episode finished! Total reward: {total_reward}")
env.close()

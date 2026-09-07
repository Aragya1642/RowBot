import gymnasium as gym

# Create agent class
class MountainCarAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95, 
    ):
        self.env = env




# Create environment
env = gym.make("MountainCar-v0", render_mode="rgb_array", goal_velocity=0.1)
print(env.action_space)

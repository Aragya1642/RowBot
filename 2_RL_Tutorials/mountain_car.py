import gymnasium as gym
from collections import defaultdict
from tqdm import tqdm

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

        # Create Q table
        self.q_values = defaultdict()

        self.lr = learning_rate
        self.discount_factor = discount_factor

        # Exploration Parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning process
        self.training_error = []

    def get_action(self) -> int:
        pass

    def update(self):
        pass

    def decay_epsilon(self):
        pass

#################################################################
# Train the agent
#################################################################
# Training hyperparameters
learning_rate = 0.001
n_episodes = 10
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1

# Create environment
env = gym.make("MountainCar-v0", render_mode="rgb_array", goal_velocity=0.1)
print(env.action_space)

# Create Agent
agent = MountainCarAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

for episode in tqdm(range(n_episodes)):
    # Reset environment
    obs, info = env.reset()
    done = False

    # Play an episode
    while not done:


        # Move to next state
        done = terminated or truncated
        obs = next_obs

    # Reduce exploration rate
    agent.decay_epsilon()

#################################################################
# Visualize training metrics
#################################################################

# TODO

#################################################################
# Test the agent
#################################################################

# TODO
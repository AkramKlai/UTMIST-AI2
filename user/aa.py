from stable_baselines3 import PPO
from user.train_agent import SB3Agent

path = "/home/penguin/Downloads/rl_model_2648700_steps.zip"

model = PPO.load(path)
env_info = model.get_env()

print(env_info)

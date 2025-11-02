from environment.environment import RenderMode, CameraResolution
from environment.agent import run_match
from stable_baselines3 import PPO
from user.train_agent import UserInputAgent, BasedAgent, ConstantAgent, ClockworkAgent, SB3Agent, RecurrentPPOAgent, CustomAgent, MLPExtractor #add anymore custom Agents (from train_agent.py) here as needed
from user.my_agent import SubmittedAgent
import pygame
pygame.init()

#my_agent = UserInputAgent()

my_agent = CustomAgent(
    sb3_class=PPO,
    file_path="/home/penguin/Downloads/rl_model_2648700_steps.zip",
    extractor=MLPExtractor
)
#my_agent = SB3Agent(file_path='/home/penguin/Downloads/UTMIST-AI2/rl_model_44475918_steps.zip')
#Input your file path here in SubmittedAgent if you are loading a model:
#opponent = SubmittedAgent()
opponent = SB3Agent(file_path='checkpoints/experiment_9/rl_model_47534034_steps.zip') # BLUE

#opponent = UserInputAgent()

match_time = 99999

# Run a single real-time match
run_match(
    agent_1=my_agent,
    agent_2=opponent,
    max_timesteps=30 * match_time,
    resolution=CameraResolution.LOW,
    video_path = "tt_agent.avi"

)
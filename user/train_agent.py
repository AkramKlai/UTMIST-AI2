'''TRAINING: AGENT

This file contains all the types of Agent classes, the Reward Function API, and the built-in train function from our multi-agent RL API for self-play training.
- All of these Agent classes are each described below. 

Running this file will initiate the training function, and will:
a) Start training from scratch
b) Continue training from a specific timestep given an input `file_path`
'''

# -------------------------------------------------------------------
# ----------------------------- IMPORTS -----------------------------
# -------------------------------------------------------------------

import torch 
import gymnasium as gym
from torch.nn import functional as F
from torch import nn as nn
import numpy as np
import pygame
from stable_baselines3 import A2C, PPO, SAC, DQN, DDPG, TD3, HER 
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from environment.agent import *
from typing import Optional, Type, List, Tuple

# -------------------------------------------------------------------------
# ----------------------------- AGENT CLASSES -----------------------------
# -------------------------------------------------------------------------

class SB3Agent(Agent):
    '''
    SB3Agent:
    - Defines an AI Agent that takes an SB3 class input for specific SB3 algorithm (e.g. PPO, SAC)
    Note:
    - For all SB3 classes, if you'd like to define your own neural network policy you can modify the `policy_kwargs` parameter in `self.sb3_class()` or make a custom SB3 `BaseFeaturesExtractor`
    You can refer to this for Custom Policy: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
    '''
    def __init__(
            self,
            sb3_class: Optional[Type[BaseAlgorithm]] = PPO,
            file_path: Optional[str] = None
    ):
        self.sb3_class = sb3_class
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = self.sb3_class("MlpPolicy", self.env, verbose=0, n_steps=30*90*3, batch_size=128, ent_coef=0.003)

            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path)

    def _gdown(self) -> str:
        # Call gdown to your link
        return

    #def set_ignore_grad(self) -> None:
        #self.model.set_ignore_act_grad(True)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )

class RecurrentPPOAgent(Agent):
    '''
    RecurrentPPOAgent:
    - Defines an RL Agent that uses the Recurrent PPO (LSTM+PPO) algorithm
    '''
    def __init__(
            self,
            file_path: Optional[str] = None
    ):
        super().__init__(file_path)
        self.lstm_states = None
        self.episode_starts = np.ones((1,), dtype=bool)

    def _initialize(self) -> None:
        if self.file_path is None:
            policy_kwargs = {
                'activation_fn': nn.ReLU,
                'lstm_hidden_size': 512,
                'net_arch': [dict(pi=[32, 32], vf=[32, 32])],
                'shared_lstm': True,
                'enable_critic_lstm': False,
                'share_features_extractor': True,

            }
            self.model = RecurrentPPO("MlpLstmPolicy",
                                      self.env,
                                      verbose=0,
                                      n_steps=30*90*20,
                                      batch_size=16,
                                      ent_coef=0.05,
                                      policy_kwargs=policy_kwargs)
            del self.env
        else:
            self.model = RecurrentPPO.load(self.file_path)

    def reset(self) -> None:
        self.episode_starts = True

    def predict(self, obs):
        action, self.lstm_states = self.model.predict(obs, state=self.lstm_states, episode_start=self.episode_starts, deterministic=True)
        if self.episode_starts: self.episode_starts = False
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    def learn(self, env, total_timesteps, log_interval: int = 2, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)

class BasedAgent(Agent):
    '''
    BasedAgent:
    - Defines a hard-coded Agent that predicts actions based on if-statements. Interesting behaviour can be achieved here.
    - The if-statement algorithm can be developed within the `predict` method below.
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.time = 0

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        action = self.act_helper.zeros()

        # If off the edge, come back
        if pos[0] > 10.67/2:
            action = self.act_helper.press_keys(['a'])
        elif pos[0] < -10.67/2:
            action = self.act_helper.press_keys(['d'])
        elif not opp_KO:
            # Head toward opponent
            if (opp_pos[0] > pos[0]):
                action = self.act_helper.press_keys(['d'])
            else:
                action = self.act_helper.press_keys(['a'])

        # Note: Passing in partial action
        # Jump if below map or opponent is above you
        if (pos[1] > 1.6 or pos[1] > opp_pos[1]) and self.time % 2 == 0:
            action = self.act_helper.press_keys(['space'], action)

        # Attack if near
        if (pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 < 4.0:
            action = self.act_helper.press_keys(['j'], action)
        return action

class UserInputAgent(Agent):
    '''
    UserInputAgent:
    - Defines an Agent that performs actions entirely via real-time player input
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.act_helper.zeros()
       
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action = self.act_helper.press_keys(['w'], action)
        if keys[pygame.K_a]:
            action = self.act_helper.press_keys(['a'], action)
        if keys[pygame.K_s]:
            action = self.act_helper.press_keys(['s'], action)
        if keys[pygame.K_d]:
            action = self.act_helper.press_keys(['d'], action)
        if keys[pygame.K_SPACE]:
            action = self.act_helper.press_keys(['space'], action)
        # h j k l
        if keys[pygame.K_h]:
            action = self.act_helper.press_keys(['h'], action)
        if keys[pygame.K_j]:
            action = self.act_helper.press_keys(['j'], action)
        if keys[pygame.K_k]:
            action = self.act_helper.press_keys(['k'], action)
        if keys[pygame.K_l]:
            action = self.act_helper.press_keys(['l'], action)
        if keys[pygame.K_g]:
            action = self.act_helper.press_keys(['g'], action)

        return action

class ClockworkAgent(Agent):
    '''
    ClockworkAgent:
    - Defines an Agent that performs sequential steps of [duration, action]
    '''
    def __init__(
            self,
            action_sheet: Optional[List[Tuple[int, List[str]]]] = None,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.steps = 0
        self.current_action_end = 0  # Tracks when the current action should stop
        self.current_action_data = None  # Stores the active action
        self.action_index = 0  # Index in the action sheet

        if action_sheet is None:
            self.action_sheet = [
                (10, ['a']),
                (1, ['l']),
                (20, ['a']),
                (3, ['a', 'j']),
                (15, ['space']),
            ]
        else:
            self.action_sheet = action_sheet

    def predict(self, obs):
        """
        Returns an action vector based on the predefined action sheet.
        """
        # Check if the current action has expired
        if self.steps >= self.current_action_end and self.action_index < len(self.action_sheet):
            hold_time, action_data = self.action_sheet[self.action_index]
            self.current_action_data = action_data  # Store the action
            self.current_action_end = self.steps + hold_time  # Set duration
            self.action_index += 1  # Move to the next action

        # Apply the currently active action
        action = self.act_helper.press_keys(self.current_action_data)
        self.steps += 1  # Increment step counter
        return action
    
class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int = 64, action_dim: int = 10, hidden_dim: int = 64):
        """
        A 3-layer MLP policy:
        obs -> Linear(hidden_dim) -> ReLU -> Linear(hidden_dim) -> ReLU -> Linear(action_dim)
        """
        super(MLPPolicy, self).__init__()

        # Input layer
        self.fc1 = nn.Linear(obs_dim, hidden_dim, dtype=torch.float32)
        # Hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
        # Output layer
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)

    def forward(self, obs):
        """
        obs: [batch_size, obs_dim]
        returns: [batch_size, action_dim]
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MLPExtractor(BaseFeaturesExtractor):
    '''
    Class that defines an MLP Base Features Extractor
    '''
    def __init__(self, observation_space: gym.Space, features_dim: int = 64, hidden_dim: int = 64):
        super(MLPExtractor, self).__init__(observation_space, features_dim)
        self.model = MLPPolicy(
            obs_dim=observation_space.shape[0], 
            action_dim=10,
            hidden_dim=hidden_dim,
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)
    
    @classmethod
    def get_policy_kwargs(cls, features_dim: int = 64, hidden_dim: int = 64) -> dict:
        return dict(
            features_extractor_class=cls,
            features_extractor_kwargs=dict(features_dim=features_dim, hidden_dim=hidden_dim) #NOTE: features_dim = 10 to match action space output
        )
    
class CustomAgent(Agent):
    def __init__(self, sb3_class: Optional[Type[BaseAlgorithm]] = PPO, file_path: str = None, extractor: BaseFeaturesExtractor = None):
        self.sb3_class = sb3_class
        self.extractor = extractor
        super().__init__(file_path)
    
    def _initialize(self) -> None:
        if self.file_path is None:
            self.model = self.sb3_class(
            "MlpPolicy",
            self.env,
            policy_kwargs=self.extractor.get_policy_kwargs(),
            verbose=0,
            n_steps=30*90*3,
            batch_size=128,
            ent_coef=0.005,       # ↓ slightly lower exploration
            learning_rate=1e-4,   # 👈 NEW: smaller LR = more stable continuation
            target_kl=0.02,       # 👈 PPO stability guard
            max_grad_norm=0.5     # 👈 prevent gradient explosions
        )

            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path)

    def _gdown(self) -> str:
        # Call gdown to your link
        return

    #def set_ignore_grad(self) -> None:
        #self.model.set_ignore_act_grad(True)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(env)
        self.model.verbose = verbose
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
        )

# --------------------------------------------------------------------------------
# ----------------------------- REWARD FUNCTIONS API -----------------------------
# --------------------------------------------------------------------------------

'''
Example Reward Functions:
- Find more [here](https://colab.research.google.com/drive/1qMs336DclBwdn6JBASa5ioDIfvenW8Ha?usp=sharing#scrollTo=-XAOXXMPTiHJ).
'''

def base_height_l2(
    env: WarehouseBrawl,
    target_height: float,
    obj_name: str = 'player'
) -> float:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # Extract the used quantities (to enable type-hinting)
    obj: GameObject = env.objects[obj_name]

    # Compute the L2 squared penalty
    return (obj.body.position.y - target_height)**2

class RewardMode(Enum):
    ASYMMETRIC_OFFENSIVE = 0
    SYMMETRIC = 1
    ASYMMETRIC_DEFENSIVE = 2

def damage_interaction_reward(
    env: WarehouseBrawl,
    mode: RewardMode = RewardMode.SYMMETRIC,
) -> float:
    """
    Computes the reward based on damage interactions between players.

    Modes:
    - ASYMMETRIC_OFFENSIVE (0): Reward is based only on damage dealt to the opponent
    - SYMMETRIC (1): Reward is based on both dealing damage to the opponent and avoiding damage
    - ASYMMETRIC_DEFENSIVE (2): Reward is based only on avoiding damage

    Args:
        env (WarehouseBrawl): The game environment
        mode (DamageRewardMode): Reward mode, one of DamageRewardMode

    Returns:
        float: The computed reward.
    """
    # Getting player and opponent from the enviornment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Reward dependent on the mode
    damage_taken = player.damage_taken_this_frame
    damage_dealt = opponent.damage_taken_this_frame

    if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
        reward = damage_dealt
    elif mode == RewardMode.SYMMETRIC:
        reward = damage_dealt - damage_taken
    elif mode == RewardMode.ASYMMETRIC_DEFENSIVE:
        reward = -damage_taken
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return reward / 140


# In[ ]:


def danger_zone_reward(
    env: WarehouseBrawl,
    zone_penalty: int = 1,
    zone_height: float = 4.2
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    reward = -zone_penalty if player.body.position.y >= zone_height else 0.0

    return reward * env.dt

def in_state_reward(
    env: WarehouseBrawl,
    desired_state: Type[PlayerObjectState]=BackDashState,
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    reward = 1 if isinstance(player.state, desired_state) else 0.0

    return reward * env.dt

def head_to_middle_reward(
    env: WarehouseBrawl,
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    multiplier = -1 if player.body.position.x > 0 else 1
    reward = multiplier * (player.body.position.x - player.prev_x)

    return reward

# def head_to_opponent(
#     env: WarehouseBrawl,
# ) -> float:

#     # Get player object from the environment
#     player: Player = env.objects["player"]
#     opponent: Player = env.objects["opponent"]

#     # Apply penalty if the player is in the danger zone
#     multiplier = -1 if player.body.position.x > opponent.body.position.x else 1
#     reward = multiplier * (player.body.position.x - player.prev_x)

#     return reward

def head_to_opponent(env: WarehouseBrawl) -> float:
    """
    Encourages the player to approach and engage the opponent intelligently.

    Key behaviors rewarded:
    - Moving closer to the opponent.
    - Facing the opponent correctly.
    - Maintaining pressure (not idling far away).
    - Slight bonus when within striking range.

    PPO-safe: rewards are clipped and scaled with env.dt.
    """

    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Current & previous horizontal positions
    x_p = float(player.body.position.x)
    x_o = float(opponent.body.position.x)
    prev_d = abs(player.prev_x - opponent.prev_x)
    curr_d = abs(x_p - x_o)

    # --- Reward approaching the opponent ---
    approaching = prev_d - curr_d       # positive if getting closer
    approach_reward = np.clip(approaching * 8.0, -1.0, 1.0)

    # --- Encourage facing the opponent ---
    facing_correctly = (
        (player.facing == 1 and x_o > x_p) or
        (player.facing == -1 and x_o < x_p)
    )
    facing_reward = 0.3 if facing_correctly else -0.3

    # --- Encourage staying near (engagement zone) ---
    engage_distance = 1.5
    if curr_d < engage_distance:
        engage_bonus = 0.5                # reward staying close
    else:
        engage_bonus = 0.0

    # --- Penalize idling far away ---
    if curr_d > 3.5 and abs(player.body.velocity.x) < 0.05:
        idle_penalty = -0.3
    else:
        idle_penalty = 0.0

    # --- Combine all ---
    reward = approach_reward + facing_reward + engage_bonus + idle_penalty

    # --- Clip & scale for PPO stability ---
    reward = np.clip(reward, -1.0, 1.0)
    return reward * env.dt


def holding_more_than_3_keys(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is holding more than 3 keys
    a = player.cur_action
    if (a > 0.5).sum() > 3:
        return env.dt
    return 0

def on_win_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return 1.0
    else:
        return -1.0

def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    else:
        return 1.0
    
def on_equip_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Hammer":
            return 1.0
        elif env.objects["player"].weapon == "Spear":
            return 0.5
    return 0.0

def on_drop_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Punch":
            return -1.0
    return 0.0

def on_combo_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == 'player':
        return -1.0
    else:
        return 1.0

'''
Add your dictionary of RewardFunctions here using RewTerms
9'''
def holding_weapon_reward(env: WarehouseBrawl) -> float:
   player = env.objects["player"]
   opponent = env.objects["opponent"]
   return 1.0 if player.weapon in ["Hammer", "Spear"] and not opponent.weapon in ["Hammer", "Spear"] else 0.0
   
def using_moving_stage_reward(env: WarehouseBrawl) -> float:
   x_pos_player = env.objects["player"].body.position[0]
   x_pos = env.objects["platform1"].body.position[0]
   return 1.0 if -1 < x_pos_player < 1 and abs(x_pos_player - x_pos) < 1 else abs(x_pos_player - x_pos) * -1.0
   
def being_higher_than_opponent_reward(env: WarehouseBrawl) -> float:
   y_pos_player = env.objects["player"].body.position[1]
   y_pos_opponent = env.objects["opponent"].body.position[1]
   return 0.5 if y_pos_player > y_pos_opponent else -0.5
   
def precise_attack_reward(
    env: WarehouseBrawl,
    success_value: float = 1.0,
    fail_value: float = -0.5,
) -> float:


    """
    Computes the reward given for every time step your agent successfully hits the opponent with their attack.


    Args:
        env (WarehouseBrawl): The game environment
        success_value (float): Reward value for the player hitting the opponent
        fail_value (float): Penalty for missing attack
    Returns:
        float: The computed reward.
    """
    reward = 0.0
    # Getting player and opponent from the enviornment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    # If the player is not in attack state, return 0 reward
    if player.state != 'attack':
        return reward
    # If the player just hit the opponent, give success reward, else give fail reward
    if opponent.just_got_hit:
        reward = success_value
    else:
        reward = fail_value
    return reward
    
def dash_avoid_attack_reward(
    env: WarehouseBrawl,
    success_value: float = 1.0,
    fail_value: float = -0.5,
) -> float:


    """
    Computes the reward given for every time step your agent successfully avoids an opponent's attack while dashing (or back dashing).


    Args:
        env (WarehouseBrawl): The game environment
        success_value (float): Reward value for the player avoiding the opponent's attack
        fail_value (float): Penalty for getting hit while dashing
    Returns:
        float: The computed reward.
    """
    reward = 0.0
    # Getting player and opponent from the enviornment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    # If the player is not dashing and opponent is not attacking, return 0 reward
    if not opponent.state == 'attack':
        return reward
    # Check if player and opponent close to each other
    distance = np.linalg.norm(player.body.position - opponent.body.position)
    if distance > 3.0:
        return reward
    # If player and opponent face each other and player is backdashing
    if player.facing != opponent.facing and player.state == 'backdash':
        reward = success_value if not player.just_got_hit else fail_value
    # if player and opponent face the same direction and player is dashing
    elif player.facing == opponent.facing and player.state == 'dash':
        reward = success_value if not player.just_got_hit else fail_value
    return reward

def aerial_attack_reward(
    env: WarehouseBrawl,
    success_value: float = 1.0,
) -> float:


    """
    Computes the reward given for every time step your agent successfully hits the opponent while in the air.


    Args:
        env (WarehouseBrawl): The game environment
        success_value (float): Reward value for the player hitting the opponent
    Returns:
        float: The computed reward.
    """
    reward = 0.0
    # Getting player and opponent from the enviornment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    if player.state == 'in-air' and opponent.just_got_hit:
        reward = success_value
    return reward
    
    
def keep_moving_reward(env) -> float:
   """
   Applies a penalty for every time frame player is not moving.
   """


   # Get player object from the environment
   player: Player = env.objects["player"]


   # Apply penalty if the player is not moving
   reward = -1 if isinstance(player.state, StandingState) else 0.0


   return reward * env.dt
   
def on_equip_reward(env: WarehouseBrawl, agent: str) -> float:
   if agent == "player":
       if env.objects["player"].weapon == "Hammer":
           return 1.0
       elif env.objects["player"].weapon == "Spear":
           return 0.5
   return 0.0
   
def weapon_grab_proximity_reward(env) -> float:
    """
    Reward being closer to weapon spawner than opponent
    (only for random spawner, not for dropped weapon spawner)

    min reward: -1
    max reward: 1
    """
    constant = 1 / (14.9 ** 2 + 9.94 ** 2) ** 0.5

    # Agent Position
    player_pos = env.objects["player"].body.position

    # Opponent Position
    opponent_pos = env.objects["opponent"].body.position

    spawners = env.get_spawner_info()
    random_spawners = []
    reward = float('-inf')
    
    for spawner in spawners:
        if spawner[0] == "Random":
            random_spawners.append(spawner)
    
    for spawner in random_spawners:
        player_distance = ((player_pos.x - spawner[1][0]) ** 2 + (player_pos.y - spawner[1][1]) ** 2) ** 0.5
        opponent_distance = ((opponent_pos.x - spawner[1][0]) ** 2 + (opponent_pos.y - spawner[1][1]) ** 2) ** 0.5
        reward = max(reward, opponent_distance - player_distance)
    
    if reward == float('-inf'):
        return 0
        
    return reward * constant

def attack_intent_reward(
    env: WarehouseBrawl,
    range_x: float = 1.6,
    range_y: float = 1.0,
    bonus: float = 0.6,
    retreat_penalty: float = -0.3,
) -> float:
    """
    Pushes the agent to actually *initiate fights* when close:
    + bonus if in attack state within melee range
    - penalty if backing away while inside melee range
    PPO-safe: clipped and scaled by env.dt externally via weight.
    """
    p: Player = env.objects["player"]
    o: Player = env.objects["opponent"]

    dx = float(abs(p.body.position.x - o.body.position.x))
    dy = float(abs(p.body.position.y - o.body.position.y))
    in_range = (dx < range_x) and (dy < range_y)

    r = 0.0
    if in_range and isinstance(p.state, AttackState):
        r += bonus
    # penalize retreat inside range (walk opposite to opponent)
    if in_range:
        moving_left = p.body.velocity.x < -0.05
        opp_on_left = o.body.position.x < p.body.position.x
        moving_right = p.body.velocity.x > 0.05
        opp_on_right = o.body.position.x > p.body.position.x
        if (moving_left and not opp_on_left) or (moving_right and not opp_on_right):
            r += retreat_penalty

    return np.clip(r, -1.0, 1.0) * env.dt


def gen_reward_manager():
    reward_functions = {
        # Survival / positioning
        'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=4.0),   # down from 4.0
        'head_to_opponent':  RewTerm(func=head_to_opponent,  weight=7.0),     # up from 3.0
        'holding_more_than_3_keys': RewTerm(func=holding_more_than_3_keys, weight=-0.2),

        # Combat (bias to offense)
        'damage_interaction_reward': RewTerm(
            func=lambda env: damage_interaction_reward(env, mode=RewardMode.ASYMMETRIC_OFFENSIVE),
            weight=5.0),                                                     # up from 2.0
        'precise_attack_reward':   RewTerm(func=precise_attack_reward,   weight=1.2),  # up from 0.8
        'dash_avoid_attack_reward':RewTerm(func=dash_avoid_attack_reward,weight=0.6),  # down from 1.0 (less defensive)
        'aerial_attack_reward':    RewTerm(func=aerial_attack_reward,    weight=1.0),

        # New: explicit attack intent near target
        'attack_intent_reward':    RewTerm(func=attack_intent_reward,    weight=2.0),

        # Tactics / resources
        'holding_weapon_reward':        RewTerm(func=holding_weapon_reward,        weight=1.5),
        'weapon_grab_proximity_reward': RewTerm(func=weapon_grab_proximity_reward, weight=0.6),  # slight downshift

        # Stage usage / movement
        'using_moving_stage_reward': RewTerm(func=using_moving_stage_reward, weight=2.0),

        # Regularization
        'penalize_attack_reward': RewTerm(func=in_state_reward, weight=-0.15, params={'desired_state': AttackState}),
        'keep_moving_reward':     RewTerm(func=keep_moving_reward, weight=0.8),
    }

    signal_subscriptions = {
        'on_win_reward':       ('win_signal',         RewTerm(func=on_win_reward, weight=15.0)),
        'on_knockout_reward':  ('knockout_signal',    RewTerm(func=on_knockout_reward, weight=10.0)),
        'on_combo_reward':     ('hit_during_stun',    RewTerm(func=on_combo_reward, weight=4.0)),  # a bit higher
        'on_equip_reward':     ('weapon_equip_signal',RewTerm(func=on_equip_reward, weight=6.0)),
        'on_drop_reward':      ('weapon_drop_signal', RewTerm(func=on_drop_reward, weight=-5.0)),
    }

    return RewardManager(reward_functions, signal_subscriptions)



# -------------------------------------------------------------------------
# ----------------------------- MAIN FUNCTION -----------------------------
# -------------------------------------------------------------------------
'''
The main function runs training. You can change configurations such as the Agent type or opponent specifications here.
'''
if __name__ == '__main__':
    # Create agent
    #my_agent = CustomAgent(sb3_class=PPO, extractor=MLPExtractor)
    my_agent = CustomAgent(file_path='checkpoints/experiment_9/rl_model_47534034_steps.zip', extractor=MLPExtractor)

    # Start here if you want to train from scratch. e.g:
    #my_agent = RecurrentPPOAgent()

    # Start here if you want to train from a specific timestep. e.g:
    #my_agent = RecurrentPPOAgent(file_path='checkpoints/experiment_3/rl_model_120006_steps.zip')

    # Reward manager
    reward_manager = gen_reward_manager()
    # Self-play settings
    selfplay_handler = SelfPlayRandom(
        partial(type(my_agent)), # Agent class and its keyword arguments
                                 # type(my_agent) = Agent class
    )

    save_handler = SaveHandler(
    agent=my_agent,
    save_freq=50_000,       # ← was 100_000, now safer and more granular
    max_saved=40,
    save_path='checkpoints',
    run_name='experiment_9',
    mode=SaveHandlerMode.RESUME
    )

    # Set opponent settings here:
    opponent_specification = {
                    'self_play': (10, selfplay_handler),
                    'constant_agent': (0.2, partial(ConstantAgent)),
                    'based_agent': (1.5, partial(BasedAgent)),
                }
    opponent_cfg = OpponentsCfg(opponents=opponent_specification)

    train(my_agent,
        reward_manager,
        save_handler,
        opponent_cfg,
        CameraResolution.LOW,
        train_timesteps=1_000_000,
        train_logging=TrainLogging.PLOT
    )
# %%
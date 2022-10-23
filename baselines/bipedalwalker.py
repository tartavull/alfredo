import os
import time
import time
import datetime

from tqdm import tqdm
import gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env


def get_time():
    value = datetime.datetime.fromtimestamp(time.time())
    return value.strftime('%Y-%m-%d_%H:%M:%S')

# Saving logs to visulise in Tensorboard, saving models
models_dir = f"/tmp/models/Mountain-{get_time()}"
logdir = f"/tmp/logs/Mountain-{get_time()}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


def get_model():
    # Parallel environments
    env = gym.make("BipedalWalker-v3")

    # The learning agent and hyperparameters
    model = PPO(
        policy=MlpPolicy,
        env=env,
        batch_size=256,
        n_steps=16,
        tensorboard_log=logdir
    )
    return model


def train(timesteps=10**6):
    model = get_model()
    for i in tqdm(range(20)): 
        model.learn(
            total_timesteps=timesteps,
            reset_num_timesteps=False, 
            progress_bar=True,
            tb_log_name="PPO")
        model.save(f"{models_dir}/{timesteps*i}")

def check_performance():
    # load the best model you observed from tensorboard 
    # the one reach the goal/ obtaining highest return
    model_path ="/tmp/models/Mountain-1666470771.6113408/80000"
    env = make_vec_env("MountainCarContinuous-v0", n_envs=1)
    best_model = PPO.load(model_path, env=env)
    print(best_model)

    obs = env.reset()
    while True:
        action, _states = best_model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

train()

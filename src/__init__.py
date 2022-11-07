from .version import __version__

import time
from copy import deepcopy

import gym
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor

def never_ending_render(env_name, model=None):
    env = gym.make(env_name)
    obs = env.reset()
    while True:
        if model:
            action, _states = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            time.sleep(1)
            obs = env.reset()
    env.close()


def train_model(config, model_constructor):
    run = wandb.init(
        project="genetic-intelligence",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    
    def make_env():
        env = gym.make(config["env_name"])
        env = Monitor(env)  # record stats such as returns
        return env
    
    if config["n_envs"] == 1:
        env = DummyVecEnv([make_env])
    else:
        env = make_vec_env(make_env, n_envs=config["n_envs"])

    recorded_env = VecVideoRecorder(
        env,
        f"/tmp/videos/{run.id}", 
        record_video_trigger=lambda x: x % (config["total_timesteps"]/20) == 0, 
        video_length=200
    )
    model = model_constructor(
        env=recorded_env,
        tensorboard_log=f"/tmp/runs/{run.id}",
        **remove_keys(config, ["env_name", "total_timesteps", "n_envs"])
    )
    model.learn(
        total_timesteps=config["total_timesteps"], 
        progress_bar=True,
        callback=WandbCallback(
            model_save_path=f"/tmp/models/{run.id}",
            verbose=2,
        ),
    )
    run.finish()
    return model

def remove_keys(d, keys):
    new_dict = deepcopy(d)
    for key in keys:
        new_dict.pop(key, None)
    return new_dict


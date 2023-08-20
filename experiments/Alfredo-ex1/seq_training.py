# ============================
# Imports
# ============================

import functools
import os
import re
from datetime import datetime

import brax
import flax
import jax
import matplotlib.pyplot as plt
import optax
from brax import envs
# from brax.envs.wrappers import training
from brax.io import html, json, model
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from jax import numpy as jp

from alfredo.agents.A0 import Alfredo
from alfredo.train import ppo


import wandb
# Initialize a new run
wandb.init(project="alfredo"
    config = {
        "env_name": "A0",
        "backend": "positional",
        "seed": 0,
        "len_training": 1_000_000,
        # add any other hyperparameters or configurations you'd like to track
    }
)

# ==============================
# Useful Functions & Data Defs
# ==============================

normalize_fn = running_statistics.normalize


def progress(num_steps, metrics):
    wandb.log({"step": num_steps, 
        "Total Reward": metrics['eval/episode_reward'],
        "Target Reward": metrics['eval/episode_reward_to_target'],
        "Vel Reward": metrics['eval/episode_reward_velocity'],
        "Alive Reward": metrics['eval/episode_reward_alive'],
        "Ctrl Reward": metrics['eval/episode_reward_ctrl'],
        "a_vel_x": metrics['eval/episode_agent_x_velocity'],
        "a_vel_y": metrics['eval/episode_agent_y_velocity']})

# ==============================
# General Variable Defs
# ==============================
cwd = os.getcwd()

# get the filepath to the scene xmls
import alfredo.scenes as scenes

scene_fp = os.path.dirname(scenes.__file__)
# ============================
# Loading and Defining Envs
# ============================
pf_paths = [f"{scene_fp}/flatworld/flatworld.xml"]

# make and save initial ppo_network
key = jax.random.PRNGKey(seed)
global_key, local_key = jax.random.split(key)
key_policy, key_value = jax.random.split(global_key)

env = Alfredo(backend=wandb.config.backend, paramFile_path=pf_paths[0])

rng = jax.random.PRNGKey(seed=1)
state = env.reset(rng)

ppo_network = ppo_networks.make_ppo_networks(
    env.observation_size, env.action_size, normalize_fn
)

init_params = ppo_losses.PPONetworkParams(
    policy=ppo_network.policy_network.init(key_policy),
    value=ppo_network.value_network.init(key_value),
)

normalizer_params = running_statistics.init_state(
    specs.Array(env.observation_size, jp.float32)
)

params_to_save = (normalizer_params, init_params.policy, init_params.value)

model.save_params(f"param-store/{env_name}_params_0", params_to_save)

# ============================
# Training & Saving Params
# ============================
i = 0

for p in pf_paths:
    # p_split_unscrper = re.split('[/_.]', p)
    # p_split_slper = re.split('[/.]', p)
    # m_name = p_split_unscrper[-3]
    # p_index = int(p_split_unscrper[-2])

    d_and_t = datetime.now()
    print(f"[{d_and_t}] loop start for model: {i}")
    env = Alfredo(backend=backend, paramFile_path=p)

    mF = f"{cwd}/param-store/{env_name}_params_{i}"
    mParams = model.load_params(mF)

    d_and_t = datetime.now()
    print(f"[{d_and_t}] jitting start for model: {i}")
    state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))
    d_and_t = datetime.now()
    print(f"[{d_and_t}] jitting end for model: {i}")

    # define new training function
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=len_training,
        num_evals=10,
        reward_scaling=0.1,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=32,
        num_updates_per_batch=8,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-3,
        num_envs=2048,
        batch_size=1024,
        seed=1,
        in_params=mParams,
    )

    d_and_t = datetime.now()
    print(f"[{d_and_t}] training start for model: {i}")
    _, params, _, ts = train_fn(environment=env, progress_fn=progress)
    d_and_t = datetime.now()
    print(f"[{d_and_t}] training end for model: {i}")

    i += 1
    next_m_name = f"param-store/{env_name}_params_{i}"
    model.save_params(next_m_name, params)

    d_and_t = datetime.now()
    print(f"[{d_and_t}] loop end for model: {i}")

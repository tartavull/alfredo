import functools
import os
import re
from datetime import datetime

import brax
import flax
import jax
import matplotlib.pyplot as plt
import optax
import wandb
from brax import envs
# from brax.envs.wrappers import training
from brax.io import html, json, model
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from jax import numpy as jp

from alfredo.agents.aant import AAnt
from alfredo.train import ppo

# Initialize a new run
wandb.init(
    project="aant",
    config={
        "env_name": "AAnt",
        "backend": "positional",
        "seed": 0,
        "len_training": 1_500_000,
        "batch_size": 1024,
    },
)

normalize_fn = running_statistics.normalize

def progress(num_steps, metrics):
    print(num_steps)
    wandb.log(
        {
            "step": num_steps,
            "Total Reward": metrics["eval/episode_reward"],
            "Lin Vel Reward": metrics["eval/episode_reward_lin_vel"],
            "Alive Reward": metrics["eval/episode_reward_alive"],
            "Ctrl Reward": metrics["eval/episode_reward_ctrl"],
            "Torque Reward": metrics["eval/episode_reward_torque"],
        }
    )

cwd = os.getcwd()

# get the filepath to the env and agent xmls
import alfredo.scenes as scenes
import alfredo.agents as agents
agents_fp = os.path.dirname(agents.__file__)
agent_xml_path = f"{agents_fp}/aant/aant.xml"

scenes_fp = os.path.dirname(scenes.__file__)

env_xml_paths = [f"{scenes_fp}/flatworld/flatworld_A1_env.xml"]

# make and save initial ppo_network
key = jax.random.PRNGKey(wandb.config.seed)
global_key, local_key = jax.random.split(key)
key_policy, key_value = jax.random.split(global_key)

env = AAnt(backend=wandb.config.backend, 
           env_xml_path=env_xml_paths[0],
           agent_xml_path=agent_xml_path)

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

model.save_params(f"param-store/AAnt_params_0", params_to_save)

# ============================
# Training & Saving Params
# ============================
i = 0

for p in env_xml_paths:

    d_and_t = datetime.now()
    print(f"[{d_and_t}] loop start for model: {i}")
    env = AAnt(backend=wandb.config.backend, 
               env_xml_path=p,
               agent_xml_path=agent_xml_path)

    mF = f"{cwd}/param-store/{wandb.config.env_name}_params_{i}"
    mParams = model.load_params(mF)

    d_and_t = datetime.now()
    print(f"[{d_and_t}] jitting start for model: {i}")
    state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))
    d_and_t = datetime.now()
    print(f"[{d_and_t}] jitting end for model: {i}")
  
    # define new training function
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=wandb.config.len_training,
        num_evals=100,
        reward_scaling=0.1,
        episode_length=1000,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=8,
        num_updates_per_batch=8,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=1e-3,
        num_envs=2048,
        batch_size=wandb.config.batch_size,
        seed=1,
        in_params=mParams,
    )

    d_and_t = datetime.now()
    print(f"[{d_and_t}] training start for model: {i}")
    _, params, _, ts = train_fn(environment=env, progress_fn=progress)
    d_and_t = datetime.now()
    print(f"[{d_and_t}] training end for model: {i}")

    i += 1
    next_m_name = f"param-store/{wandb.config.env_name}_params_{i}"
    model.save_params(next_m_name, params)

    d_and_t = datetime.now()
    print(f"[{d_and_t}] loop end for model: {i}")

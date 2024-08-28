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
        "seed": 1,
        "len_training": 1_500_000,
        "num_evals": 500,
        "num_envs": 2048,
        "batch_size": 2048,
        "num_minibatches": 8,
        "updates_per_batch": 8,
        "episode_len": 1000,
        "unroll_len": 10,
        "reward_scaling":0.1,
        "action_repeat": 1,
        "discounting": 0.97,
        "learning_rate": 3e-4,
        "entropy_cost": 1e-3,
        "reward_scaling": 0.1,
        "normalize_obs": True,
    },
)

normalize_fn = running_statistics.normalize

def progress(num_steps, metrics):
    print(num_steps)
    print(metrics)
    epi_len = wandb.config.episode_len
    wandb.log(
        {
            "step": num_steps,
            "Total Reward": metrics["eval/episode_reward"]/epi_len,
            "Waypoint Reward": metrics["eval/episode_reward_waypoint"]/epi_len,
            #"Lin Vel Reward": metrics["eval/episode_reward_lin_vel"],
            #"Yaw Vel Reward": metrics["eval/episode_reward_yaw_vel"],
            "Alive Reward": metrics["eval/episode_reward_alive"]/epi_len,
            "Ctrl Reward": metrics["eval/episode_reward_ctrl"]/epi_len,
            "Upright Reward": metrics["eval/episode_reward_upright"]/epi_len,
            "Torque Reward": metrics["eval/episode_reward_torque"]/epi_len,
            "Abs Pos X World": metrics["eval/episode_pos_x_world_abs"]/epi_len, 
            "Abs Pos Y World": metrics["eval/episode_pos_y_world_abs"]/epi_len, 
            "Abs Pos Z World": metrics["eval/episode_pos_z_world_abs"]/epi_len,
            "Dist Goal X": metrics["eval/episode_dist_goal_x"]/epi_len, 
            "Dist Goal Y": metrics["eval/episode_dist_goal_y"]/epi_len, 
            #"Dist Goal Z": metrics["eval/episode_dist_goal_z"]/epi_len,
        }
    )

cwd = os.getcwd()

# get the filepath to the env and agent xmls
import alfredo.scenes as scenes
import alfredo.agents as agents
agents_fp = os.path.dirname(agents.__file__)
agent_xml_path = f"{agents_fp}/aant/aant.xml"

scenes_fp = os.path.dirname(scenes.__file__)

env_xml_paths = [f"{scenes_fp}/flatworld/flatworld_A1_env.xml", 
                 f"{scenes_fp}/flatworld/flatworld_A1_env.xml",
                 f"{scenes_fp}/flatworld/flatworld_A1_env.xml",
                 f"{scenes_fp}/flatworld/flatworld_A1_env.xml"]

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
i = 8

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
    state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=1))
    d_and_t = datetime.now()
    print(f"[{d_and_t}] jitting end for model: {i}")
  
    # define new training function
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=wandb.config.len_training,
        num_evals=wandb.config.num_evals,
        reward_scaling=wandb.config.reward_scaling,
        episode_length=wandb.config.episode_len,
        normalize_observations=wandb.config.normalize_obs,
        action_repeat=wandb.config.action_repeat,
        unroll_length=wandb.config.unroll_len,
        num_minibatches=wandb.config.num_minibatches,
        num_updates_per_batch=wandb.config.updates_per_batch,
        discounting=wandb.config.discounting,
        learning_rate=wandb.config.learning_rate,
        entropy_cost=wandb.config.entropy_cost,
        num_envs=wandb.config.num_envs,
        batch_size=wandb.config.batch_size,
        seed=wandb.config.seed,
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

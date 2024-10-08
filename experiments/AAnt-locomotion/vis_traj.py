import functools
import os
import re
import sys
from datetime import datetime

import brax
import jax
import matplotlib.pyplot as plt
from brax import envs
from brax.envs.wrappers import training
from brax.io import html, json, model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from jax import numpy as jp

from alfredo.agents.aant import AAnt

from alfredo.rewards import Reward
from alfredo.rewards import rTracking_lin_vel
from alfredo.rewards import rTracking_yaw_vel

backend = "positional"

# Load desired model xml and trained param set
# get filepaths from commandline args
cwd = os.getcwd()

# get the filepath to the env and agent xmls
import alfredo.scenes as scenes
import alfredo.agents as agents
agents_fp = os.path.dirname(agents.__file__)
agent_xml_path = f"{agents_fp}/aant/aant.xml"

scenes_fp = os.path.dirname(scenes.__file__)

env_xml_path = f"{scenes_fp}/{sys.argv[-2]}"
tpf_path = f"{cwd}/{sys.argv[-1]}"

print(f"agent description file: {agent_xml_path}")
print(f"environment description file: {env_xml_path}")
print(f"neural parameter file: {tpf_path}")

params = model.load_params(tpf_path)

# Define Reward Structure
# For visualizing, this is just to be able to create the env
# May want to make this not necessary in the future ..?
rewards = {'r_lin_vel': Reward(rTracking_lin_vel, sc=8.0, ps={}),
           'r_yaw_vel': Reward(rTracking_yaw_vel, sc=1.0, ps={})}

# create an env with auto-reset and load previously trained parameters
env = AAnt(backend=backend,
           rewards=rewards, 
           env_xml_path=env_xml_path,
           agent_xml_path=agent_xml_path)

auto_reset = True
episode_length = 1000
action_repeat = 1

#if episode_length is not None:
#    env = training.EpisodeWrapper(env, episode_length, action_repeat)

#if auto_reset:
#    env = training.AutoResetWrapper(env)

jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)

rollout = []
rng = jax.random.PRNGKey(seed=13194)
state = jit_env_reset(rng=rng)

normalize = lambda x, y: x
normalize = running_statistics.normalize

ppo_network = ppo_networks.make_ppo_networks(
    state.obs.shape[-1], env.action_size, preprocess_observations_fn=normalize
)

make_policy = ppo_networks.make_inference_fn(ppo_network)
policy_params = (params[0], params[1])
inference_fn = make_policy(policy_params)

jit_inference_fn = jax.jit(inference_fn)

x_vel = 0.0     # m/s
y_vel = 3.0     # m/s
yaw_vel = 0.0   # rad/s
jcmd = jp.array([x_vel, y_vel, yaw_vel])

wcmd = jp.array([10.0, 10.0])

# generate policy rollout
for _ in range(episode_length):
    rollout.append(state.pipeline_state)
    act_rng, rng = jax.random.split(rng)

    state.info['jcmd'] = jcmd
    state.info['wcmd'] = wcmd
    act, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_env_step(state, act)
    print(state.info)


print(rollout[-1])

html_string = html.render(env.sys.replace(dt=env.dt), rollout)

# save output to html filepaths
d_and_t = datetime.now()
d_and_t = d_and_t.strftime("%Y-%m-%d_%H-%M-%S")
tpf_path_split = re.split("[/]", tpf_path)
html_file_path = f"{cwd}/vis-store/{tpf_path_split[-1]}_{d_and_t}.html"
html_file_path = html_file_path.replace(" ", "_")

with open(html_file_path, "w") as file:
    file.write(html_string)
    print(f"saved visualization to {html_file_path}")

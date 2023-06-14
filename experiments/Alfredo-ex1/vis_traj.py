import functools
import jax
import os
import re
import sys

from datetime import datetime
from jax import numpy as jp
import matplotlib.pyplot as plt

import brax

from brax import envs
from brax.io import model
from brax.io import json
from brax.io import html
from brax.training.agents.ppo import networks as ppo_networks
from brax.envs.wrappers import training
from brax.training.acme import running_statistics

from alfredo.agents.A0.alfredo_0 import Alfredo

backend = 'positional'

# Load desired model xml and trained param set
# get filepaths from commandline args
cwd = os.getcwd()

import alfredo.scenes as scenes
scene_fp = os.path.dirname(scenes.__file__)

pf_path = f"{scene_fp}/{sys.argv[-2]}"
tpf_path = f"{cwd}/{sys.argv[-1]}"
params = model.load_params(tpf_path)

# create an env with auto-reset and load previously trained parameters
env = Alfredo(backend=backend, paramFile_path=pf_path)

auto_reset = True
episode_length = 1000
action_repeat = 1

if episode_length is not None:
  env = training.EpisodeWrapper(env, episode_length, action_repeat)

if auto_reset:
  env = training.AutoResetWrapper(env)

jit_env_reset = jax.jit(env.reset)
jit_env_step = jax.jit(env.step)

rollout = []
rng = jax.random.PRNGKey(seed=1)
state = jit_env_reset(rng=rng)

normalize = lambda x, y: x
normalize = running_statistics.normalize

ppo_network = ppo_networks.make_ppo_networks(
        state.obs.shape[-1],
        env.action_size,
        preprocess_observations_fn=normalize)

make_policy = ppo_networks.make_inference_fn(ppo_network)
policy_params = (params[0], params[1])
inference_fn = make_policy(policy_params)

jit_inference_fn = jax.jit(inference_fn)

# generate policy rollout
for _ in range(episode_length):
  rollout.append(state.pipeline_state)
  act_rng, rng = jax.random.split(rng)
  act, _ = jit_inference_fn(state.obs, act_rng)
  state = jit_env_step(state, act)

html_string = html.render(env.sys.replace(dt=env.dt), rollout)

# save output to html filepaths
d_and_t = datetime.now()
tpf_path_split = re.split('[/]', tpf_path)
html_file_path = f'{cwd}/vis-store/{tpf_path_split[-1]}_{d_and_t}.html'
html_file_path = html_file_path.replace(' ', '_')

with open(html_file_path, "w") as file:
    file.write(html_string)
    print(f"saved visualization to {html_file_path}")

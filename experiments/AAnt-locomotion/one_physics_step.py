import functools
import os
import re
import sys
from datetime import datetime

import brax
import jax
import matplotlib.pyplot as plt
from brax import envs, math
from brax.envs.wrappers import training
from brax.io import html, json, model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from jax import numpy as jp

from alfredo.agents.aant import AAnt

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

env_xml_paths = [f"{scenes_fp}/flatworld/flatworld_A1_env.xml"]

# create an env and initial state
env = AAnt(backend=backend, 
           env_xml_path=env_xml_paths[0],
           agent_xml_path=agent_xml_path)

state = env.reset(rng=jax.random.PRNGKey(seed=0))
#state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))

x_vel = 0.8     # m/s
y_vel = 0.0     # m/s
yaw_vel = 0.0   # rad/s
jcmd = jp.array([x_vel, y_vel, yaw_vel])
state.info['jcmd'] = jcmd

#up = jp.array([0.0, 0.0, 1])
# rot_up = math.rotate(up, jp.array([1, 0, 0, 0]))
#rot_up = math.rotate(up, state.pipeline_state.x.rot[6])
#rew = jp.dot(up, rot_up)
#print(f"x.rot = {state.pipeline_state.x.rot}")
#print(f"up: {up}, rot_up: {rot_up}")
#print(rew)

print(f"\n-----------------------------------------------------------------\n")
state = env.step(state, jp.zeros(env.action_size))

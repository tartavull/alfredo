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

from alfredo.agents.A1.alfredo_1 import Alfredo

backend = "positional"

# Load desired model xml and trained param set
# get filepaths from commandline args
cwd = os.getcwd()

# get the filepath to the env and agent xmls
import alfredo.scenes as scenes

import alfredo.agents as agents
agents_fp = os.path.dirname(agents.__file__)
agent_xml_path = f"{agents_fp}/A1/a1.xml"

scenes_fp = os.path.dirname(scenes.__file__)

env_xml_paths = [f"{scenes_fp}/flatworld/flatworld_A1_env.xml"]

# create an env and initial state
env = Alfredo(backend=backend, 
              env_xml_path=env_xml_paths[0],
              agent_xml_path=agent_xml_path)

state = env.reset(rng=jax.random.PRNGKey(seed=0))
#state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))

x_vel = 0.8     # m/s
y_vel = 0.0     # m/s
yaw_vel = 0.0   # rad/s
jcmd = jp.array([x_vel, y_vel, yaw_vel])
state.info['jcmd'] = jcmd

#print(f"Alfredo brax env dir: {dir(env)}")
#print(f"state: {state}")

com, *_ = env._com(state.pipeline_state)
obs = env._get_obs(state.pipeline_state, jp.zeros(env.action_size))
#print(f"CoM = {com}")
#print(f"pipeline_state: {state.pipeline_state}")
#print(f"observation: {obs}")
print(f"\n-----------------------------------------------------------------\n")
state = env.step(state, jp.zeros(env.action_size))
com, *_ = env._com(state.pipeline_state)
obs = env._get_obs(state.pipeline_state, jp.zeros(env.action_size))
#print(f"CoM = {com}")
#print(f"pipeline_state: {state.pipeline_state}")
#print(f"observation: {obs}")

print(f"\n-----------------------------------------------------------------\n")
state = env.step(state, jp.zeros(env.action_size))
com, *_ = env._com(state.pipeline_state)
obs = env._get_obs(state.pipeline_state, jp.zeros(env.action_size))
#print(f"CoM = {com}")
#print(f"pipeline_state: {state.pipeline_state}")
#print(f"observation: {obs}")

print(f"\n-----------------------------------------------------------------\n")
state = env.step(state, jp.zeros(env.action_size))
com, *_ = env._com(state.pipeline_state)
obs = env._get_obs(state.pipeline_state, jp.zeros(env.action_size))
#print(f"CoM = {com}")
#print(f"pipeline_state: {state.pipeline_state}")
#print(f"observation: {obs}")

print(f"\n-----------------------------------------------------------------\n")
state = env.step(state, jp.zeros(env.action_size))
com, *_ = env._com(state.pipeline_state)
obs = env._get_obs(state.pipeline_state, jp.zeros(env.action_size))
#print(f"CoM = {com}")
#print(f"pipeline_state: {state.pipeline_state}")
#print(f"observation: {obs}")

print(f"\n-----------------------------------------------------------------\n")
state = env.step(state, jp.zeros(env.action_size))
com, *_ = env._com(state.pipeline_state)
obs = env._get_obs(state.pipeline_state, jp.zeros(env.action_size))
#print(f"CoM = {com}")
#print(f"pipeline_state: {state.pipeline_state}")
#print(f"observation: {obs}")



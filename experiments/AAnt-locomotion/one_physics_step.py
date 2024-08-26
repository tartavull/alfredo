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

tpf_path = f"{cwd}/{sys.argv[-1]}"

print(f"agent description file: {agent_xml_path}")
print(f"environment description file: {env_xml_paths[0]}")
print(f"neural parameter file: {tpf_path}")

params = model.load_params(tpf_path)

# create an env and initial state
env = AAnt(backend=backend, 
           env_xml_path=env_xml_paths[0],
           agent_xml_path=agent_xml_path)

rng = jax.random.PRNGKey(seed=3)
state = env.reset(rng=rng) #state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))

normalize = lambda x, y: x
normalize = running_statistics.normalize

ppo_network = ppo_networks.make_ppo_networks(
    state.obs.shape[-1], env.action_size, preprocess_observations_fn=normalize
)

make_policy = ppo_networks.make_inference_fn(ppo_network)
policy_params = (params[0], params[1])
inference_fn = make_policy(policy_params)

wcmd = jp.array([0.0, 1000.0])
key_envs, _ = jax.random.split(rng)
state = env.reset(rng=key_envs)

print(f"q: {state.pipeline_state.q}")
print(f"\n")
print(f"qd: {state.pipeline_state.qd}")
print(f"\n")
print(f"x: {state.pipeline_state.x}")
print(f"\n")
print(f"xd: {state.pipeline_state.xd}")
print(f"\n")
print(f"contact: {state.pipeline_state.contact}")
print(f"\n")
print(f"reward: {state.reward}")
print(f"\n")
print(state.metrics)
print(f"\n")
print(f"done: {state.done}")
print(f"\n")

episode_length = 10
for _ in range(episode_length):
    print(f"\n---------------------------------------------------------------\n")
    
    act_rng, rng = jax.random.split(rng)
    print(f"rng: {rng}")    
    print(f"act_rng: {act_rng}")
    print(f"\n")
    
    state.info['wcmd'] = wcmd
    print(f"state info: {state.info}")
    print(f"\n")

    act, _ = inference_fn(state.obs, act_rng)
    print(f"observation: {state.obs}")
    print(f"\n")
    print(f"action: {act}")
    print(f"\n")

    state = env.step(state, act)

    print(f"q: {state.pipeline_state.q}")
    print(f"\n")
    print(f"qd: {state.pipeline_state.qd}")
    print(f"\n")
    print(f"x: {state.pipeline_state.x}")
    print(f"\n")
    print(f"xd: {state.pipeline_state.xd}")
    print(f"\n")
    print(f"contact: {state.pipeline_state.contact}")
    print(f"\n")
    print(f"reward: {state.reward}")
    print(f"\n")
    print(state.metrics)
    print(f"\n")
    print(f"done: {state.done}")
    print(f"\n")

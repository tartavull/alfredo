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

import alfredo.scenes as scenes

scene_fp = os.path.dirname(scenes.__file__)
pf_path = f"{scene_fp}/{sys.argv[-1]}"

# create an env and initial state
env = Alfredo(backend=backend, paramFile_path=pf_path)
state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))

# render scene
html_string = html.render(env.sys.replace(dt=env.dt), [state.pipeline_state])

# save output to html filepaths
d_and_t = datetime.now()
html_file_path = f"{cwd}/vis-store/A1_dev_.html"

html_file_path = html_file_path.replace(" ", "_")

with open(html_file_path, "w") as file:
    file.write(html_string)
    print(f"saved visualization to {html_file_path}")

import functools
import os
import re
import sys
import importlib
import inspect

import brax
import jax
from brax import envs
from brax.envs.base import PipelineEnv
from brax.base import State, System
from brax.io import html, json, model
from jax import numpy as jp

from alfredo.agents import *

def generate_wiggle_traj(env: PipelineEnv, dt=0.1, motion_time=1.0):
    """
    Generate html visual of wiggle trajectory.
    Primarily used for debugging new models

    Parameters:
    - env (PipelineEnv):
    - dt (float): The time step duration for which each action is applied.
    - motion_time (float): The total time duration for jogging from -1 to 1.

    Returns:
    - HTML string
    """

    # Generate Wiggle     
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)

    rollout = []
    rng = jax.random.PRNGKey(seed=0)
    state = jit_env_reset(rng=rng)

    wiggle_actions = generate_wiggle_actions(env.action_size, dt, motion_time)

    for wa in wiggle_actions:
        print(f"commanding: {wa}")
        rollout.append(state.pipeline_state)
        act_rng, rng = jax.random.split(rng)

        state = jit_env_step(state, wa) 
    
    
    traj_html_str = html.render(env.sys.replace(dt=env.dt), rollout)
    
    return traj_html_str

def generate_wiggle_actions(action_size, dt=0.1, motion_time=1.0):
    """
    Generate action vectors to gradually jog each actuator from 
    -1 to 1 (normalized control values).

    Parameters:
    - action_size (int): The number of actuators in the model.
    - dt (float): The time step duration for which each action is applied.
    - motion_time (float): The total time duration for jogging from -1 to 1.

    Returns:
    - List of action vectors for jogging each actuator.
    """

    actions = []

    # Calculate the number of steps required for the full jog
    total_steps = int(motion_time / dt)
    
    # Calculate the increment based on the total steps
    increment = 2.0 / total_steps  # Since we are jogging from -1 to 1

    # Generate action sequences for each actuator
    for i in range(action_size):
        # Jogging forward
        for j in range(total_steps):
            action_vector = jp.zeros(action_size)
            action_vector = action_vector.at[i].set(-1.0 + increment * (j + 1))  # Gradual increase
            actions.append(action_vector)

        # Jogging backward
        for j in range(total_steps):
            action_vector = jp.zeros(action_size)
            action_vector = action_vector.at[i].set(1.0 - increment * (j + 1))  # Gradual decrease
            actions.append(action_vector)

    return actions

if __name__ == '__main__':

    backend = "positional"

    # Load desired model xml and trained param set
    # get filepaths from commandline args
    cwd = os.getcwd()

    # Get the filepath to the env and agent xmls
    import alfredo.scenes as scenes
    import alfredo.agents as agents

    agent_name = sys.argv[-2]
    module_name = f"alfredo.agents.{agent_name}"
    
    agents_fp = os.path.dirname(agents.__file__)
    agent_xml_path = f"{agents_fp}/{agent_name}/{agent_name}.xml"

    scenes_fp = os.path.dirname(scenes.__file__)
    env_xml_path = f"{scenes_fp}/{sys.argv[-1]}"

    print(f"agent description file: {agent_xml_path}")
    print(f"environment description file: {env_xml_path}")

    # Find & create Agent Brax environment
    env_init_params = {"backend": backend,
                       "env_xml_path": env_xml_path,
                       "agent_xml_path": agent_xml_path}

    module = importlib.import_module(module_name)
    
    classes_in_module = [member for name, member in inspect.getmembers(module, inspect.isclass) 
                         if member.__module__.startswith(module.__name__)]

    if len(classes_in_module) == 1:
        agentClass = classes_in_module[0]
        env = agentClass(**env_init_params)
    else:
        raise ImportError(f"Agent Class not Found")

    traj_html_str = generate_wiggle_traj(env, dt=env.dt)

    cwd = os.getcwd()
    save_fp = f"{cwd}/vis-store/{agent_name}_wiggle_traj.html"
    save_fp = save_fp.replace(" ", "_")

    with open(save_fp, "w") as file:
        file.write(traj_html_str)
        print(f"saved wiggle traj visualization to {save_fp}")
     

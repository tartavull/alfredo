from typing import Tuple

import jax
from brax import actuator, base, math
from brax.envs import PipelineEnv, State
from brax.io import mjcf
from etils import epath
from jax import numpy as jp

def rSpeed_X(sys: base.System,
             pipeline_state: base.State,
             CoM_prev: jp.ndarray,
             CoM_now: jp.ndarray,
             dt,
             weight=1.0,
             focus_idx_range=(0, -1)) -> jp.ndarray:
   

    velocity = (CoM_now - CoM_prev) / dt
    
    focus_s = focus_idx_range[0]
    focus_e = focus_idx_range[-1]
    
    sxr = weight * velocity[0]

    return jp.array([sxr, velocity[0], velocity[1]])

def rSpeed_Y(sys: base.System,
             pipeline_state: base.State,
             CoM_prev: jp.ndarray,
             CoM_now: jp.ndarray,
             dt,
             weight=1.0,
             focus_idx_range=(0, -1)) -> jp.ndarray:
   

    velocity = (CoM_now - CoM_prev) / dt
    
    focus_s = focus_idx_range[0]
    focus_e = focus_idx_range[-1]
    
    syr = weight * velocity[1]

    return jp.array([syr, velocity[0], velocity[1]])

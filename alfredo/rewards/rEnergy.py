from typing import Tuple

import jax
from brax import actuator, base, math
from brax.envs import PipelineEnv, State
from brax.io import mjcf
from etils import epath
from jax import numpy as jp


def rTorques(sys: base.System, 
             pipeline_state: base.State, 
             action: jp.ndarray,
             weight=1.0,
             focus_idx_range=(0, -1)) -> jp.ndarray:
    
    s_idx = focus_idx_range[0]
    e_idx = focus_idx_range[1]
    
    torque = actuator.to_tau(sys,
                             action,
                             pipeline_state.q[s_idx:e_idx],
                             pipeline_state.qd[s_idx:e_idx])
    
    
    tr = jp.sqrt(jp.sum(jp.square(torque))) + jp.sum(jp.abs(torque))

    return weight*tr

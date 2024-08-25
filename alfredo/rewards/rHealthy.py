from typing import Tuple

import jax
from brax import actuator, base, math
from brax.envs import PipelineEnv, State
from brax.io import mjcf
from etils import epath
from jax import numpy as jp

def rHealthy_simple_z(sys: base.System, 
                      pipeline_state: base.State, 
                      z_range: Tuple,
                      early_terminate: True,
                      weight=1.0,
                      focus_idx_range=(0, -1)) -> jp.ndarray:

    min_z, max_z = z_range
    focus_s = focus_idx_range[0]
    focus_e = focus_idx_range[-1]
    
    focus_x_pos = pipeline_state.x.pos[focus_s, focus_e]

    is_healthy = jp.where(focus_x_pos < min_z, x=0.0, y=1.0)
    is_healthy = jp.where(focus_x_pos > max_z, x=0.0, y=is_healthy)

    if early_terminate:
        hr = weight
    else:
        hr = weight * is_healthy

    return jp.array([hr, is_healthy])

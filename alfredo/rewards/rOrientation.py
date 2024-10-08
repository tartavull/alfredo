from typing import Tuple

import jax
from brax import actuator, base, math
from brax.envs import PipelineEnv, State
from brax.io import mjcf
from etils import epath
from jax import numpy as jp

def rUpright(sys: base.System,
             pipeline_state: base.State,
             focus_idx_range = (0,0)) -> jax.Array:

    up = jp.array([0.0, 0.0, 1.0])
    rot_up = math.rotate(up, pipeline_state.x.rot[0])
    
    return jp.dot(up, rot_up)

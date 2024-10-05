from typing import Tuple

import jax
from brax import actuator, base, math
from brax.envs import PipelineEnv, State
from brax.io import mjcf
from etils import epath
from jax import numpy as jp

def rConstant(sys: base.System, 
              pipeline_state: base.State,
              focus_idx_range=(0, -1)) -> jax.Array:
   
    return jp.array([1.0])

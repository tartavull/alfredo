from typing import Tuple

import jax
from brax import actuator, base, math
from brax.envs import PipelineEnv, State
from brax.io import mjcf
from etils import epath
from jax import numpy as jp

def rControl_act_ss(sys: base.System, 
                    pipeline_state: base.State,
                    action: jp.ndarray,
                    weight=1.0,
                    focus_idx_range=(1, -1)) -> jp.ndarray: 

    ctrl_cost = weight * jp.sum(jp.square(action))
    
    return ctrl_cost 

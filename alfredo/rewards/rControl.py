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
                    focus_idx_range=(0, -1)) -> jax.Array: 

    ctrl_cost = weight * jp.sum(jp.square(action))
    
    return jp.array([ctrl_cost]) 

def rTracking_lin_vel(sys: base.System,
                     pipeline_state: base.State,
                     jcmd: jax.Array,
                     sigma=0.25,
                     focus_idx_range=(0, 1)) -> jax.Array:

    local_vel = math.rotate(pipeline_state.xd.vel[0], 
                            math.quat_inv(pipeline_state.x.rot[0]))

    lv_error = jp.sum(jp.square(jcmd[:2] - local_vel[:2])) # just taking a look at x, y velocities
    lv_reward = jp.exp(-lv_error/sigma)
    
    return jp.array([lv_reward])

def rTracking_yaw_vel(sys: base.System,
                      pipeline_state: base.State,
                      jcmd: jax.Array,
                      sigma=0.25,
                      focus_idx_range=(0, 1)) -> jax.Array:

    local_ang_vel = math.rotate(pipeline_state.xd.ang[0], 
                                math.quat_inv(pipeline_state.x.rot[0]))

    yaw_vel_error = jp.square(jcmd[2] - local_ang_vel[2])
    yv_reward = jp.exp(-yaw_vel_error/sigma)

    return jp.array([yv_reward])


def rTracking_Waypoint(sys: base.System,
                       pipeline_state: base.State,
                       wcmd: jax.Array,
                       focus_idx_range=0) -> jax.Array:

    torso_pos = pipeline_state.x.pos[focus_idx_range]
    pos_goal_diff = torso_pos[0:2] - waypoint[0:2] 
    pos_sum_abs_diff = -jp.sum(jp.abs(pos_goal_diff))
    
    return jp.array([pos_sum_abs_diff])

def rStand_still(sys: base.System,
                 pipeline_state: base.State,
                 jcmd: jax.Array,
                 joint_angles: jax.Array,
                 default_pose: jax.Array,
                 focus_idx_range=0) -> jax.Array:

    close_to_still = jp.sum(jp.abs(joint_angles - default_pose)) * math.normalize(jcmd[:2])[1] < 0.1

    return jp.array([close_to_still])

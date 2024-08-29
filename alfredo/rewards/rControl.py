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
                    focus_idx_range=(0, -1)) -> jp.ndarray: 

    ctrl_cost = weight * jp.sum(jp.square(action))
    
    return ctrl_cost 

def rTracking_lin_vel(sys: base.System,
                     pipeline_state: base.State,
                     CoM_prev: jp.ndarray,
                     CoM_now: jp.ndarray,
                     dt,
                     jcmd: jax.Array,
                     weight=1.0,
                     sigma=0.25,
                     focus_idx_range=(0, 1)) -> jp.ndarray:

    #local_vel = math.rotate(pipeline_state.xd.vel[focus_idx_range[0]:focus_idx_range[1]], 
    #                        math.quat_inv(pipeline_state.x.rot[focus_idx_range[0]:focus_idx_range[1]]))
    local_vel = math.rotate(pipeline_state.xd.vel[0], 
                            math.quat_inv(pipeline_state.x.rot[0]))

    #print(f"com_prev:{CoM_prev}, com_now:{CoM_now}")
    #local_vel = (CoM_prev - CoM_now)/dt
    #print(f"jcmd[:2]: {jcmd[:2]}")
    #print(f"loca_vel[:2]: {local_vel[:2]}")
    lv_error = jp.sum(jp.square(jcmd[:2] - local_vel[:2])) # just taking a look at x, y velocities
    lv_reward = jp.exp(-lv_error/sigma)
    
    #print(f"lv_error: {lv_error}")
    #print(f"lv_reward: {lv_reward}")

    return weight*lv_reward

def rTracking_yaw_vel(sys: base.System,
                      pipeline_state: base.State,
                      jcmd: jax.Array,
                      sigma=0.25,
                      weight=1.0,
                      focus_idx_range=(0, 1)) -> jp.ndarray:

    #local_yaw_vel = math.rotate(pipeline_state.xd.ang[focus_idx_range[0]:focus_idx_range[1]], 
    #                            math.quat_inv(pipeline_state.x.rotate[focus_idx_range[0], focus_idx_range[1]])) 
    local_yaw_vel = math.rotate(pipeline_state.xd.vel[0], 
                                math.quat_inv(pipeline_state.x.rot[0]))
    yaw_vel_error = jp.square(jcmd[2] - local_yaw_vel[2])
    yv_reward = jp.exp(-yaw_vel_error/sigma)

    return weight*yv_reward


def rTracking_Waypoint(sys: base.System,
                       pipeline_state: base.State,
                       waypoint: jax.Array,
                       weight=1.0,
                       focus_idx_range=0) -> jp.ndarray:

    # x_i = pipeline_state.x.vmap().do(
    #     base.Transform.create(pos=sys.link.inertia.transform.pos)
    # )

    #print(f"wcmd: {waypoint}")
    #print(f"x.pos[0]: {pipeline_state.x.pos[0]}")
    torso_pos = pipeline_state.x.pos[focus_idx_range]
    pos_goal_diff = torso_pos[0:2] - waypoint[0:2] 
    #print(f"pos_goal_diff: {pos_goal_diff}")
    pos_sum_abs_diff = -jp.sum(jp.abs(pos_goal_diff))
    #inv_euclid_dist = -math.safe_norm(pos_goal_diff)
    #print(f"pos_sum_abs_diff: {pos_sum_abs_diff}") 
    
    #return weight*inv_euclid_dist
    return weight*pos_sum_abs_diff

def rStand_still(sys: base.System,
                 pipeline_state: base.State,
                 jcmd: jax.Array,
                 joint_angles, jax.Array
                 default_pose, jax.Array
                 weight: 1.0,
                 focus_idx_range=0i,) -> jp.ndarray:

    close_to_still = jp.sum(jp.abs(joint_angles - default_pose)) * math.normalize(jcmd[:2])[1] < 0.1

    return weight * close_to_still

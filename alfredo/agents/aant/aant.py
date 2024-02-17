from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp

from alfredo.tools import compose_scene
from alfredo.rewards import rConstant
from alfredo.rewards import rHealthy_simple_z
from alfredo.rewards import rSpeed_X
from alfredo.rewards import rControl_act_ss
from alfredo.rewards import rTorques
from alfredo.rewards import rTracking_lin_vel
from alfredo.rewards import rTracking_yaw_vel
from alfredo.rewards import rUpright
from alfredo.rewards import rTracking_Waypoint

class AAnt(PipelineEnv):
    """ """

    def __init__(self,
                 ctrl_cost_weight=0.5,
                 use_contact_forces=False,
                 contact_cost_weight=5e-4,
                 healthy_reward=1.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.4, 1.0),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 backend='generalized',
                 **kwargs,):

        # forcing this model to need an input scene_xml_path or 
        # the combination of env_xml_path and agent_xml_path
        # if none of these options are present, an error will be thrown
        path=""

        if "env_xml_path" and "agent_xml_path" in kwargs: 
            env_xp = kwargs["env_xml_path"]
            agent_xp = kwargs["agent_xml_path"]
            xml_scene = compose_scene(env_xp, agent_xp)
            del kwargs["env_xml_path"]
            del kwargs["agent_xml_path"]
            
            sys = mjcf.loads(xml_scene)
        

        n_frames = 5

        if backend in ['spring', 'positional']:
            sys = sys.replace(dt=0.005)
            n_frames = 10 

        if backend == 'positional':
            # TODO: does the same actuator strength work as in spring
            sys = sys.replace(
                actuator=sys.actuator.replace(
                    gear=200 * jp.ones_like(sys.actuator.gear)
                )
            )

        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._use_contact_forces = use_contact_forces
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if self._use_contact_forces:
            raise NotImplementedError('use_contact_forces not implemented.')        


    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale

        jcmd = self._sample_command(rng3)
        wcmd = self._sample_waypoint(rng3) 

        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        state_info = {
            'jcmd':jcmd,
            'wcmd':wcmd,
        }
        
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state, state_info)

        reward, done, zero = jp.zeros(3)
        metrics = {
            'reward_ctrl': zero,
            'reward_alive': zero,
            'reward_torque': zero,
            #'reward_lin_vel': zero,
            #'reward_yaw_vel': zero,
            'reward_upright': zero,
            'reward_waypoint': zero,
        }

        return State(pipeline_state, obs, reward, done, metrics, state_info)

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
                
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        waypoint_cost = rTracking_Waypoint(self.sys,
                                           state.pipeline_state,
                                           state.info['wcmd'],
                                           weight=1.0,
                                           focus_idx_range=0)

        lin_vel_reward = rTracking_lin_vel(self.sys,
                                           state.pipeline_state,
                                           jp.array([0, 0, 0]), #dummy values for previous CoM
                                           jp.array([0, 0, 0]), #dummy values for current CoM
                                           self.dt,
                                           state.info['jcmd'],
                                           weight=15.5,
                                           focus_idx_range=(0,0))

        yaw_vel_reward = rTracking_yaw_vel(self.sys,
                                           state.pipeline_state,
                                           state.info['jcmd'],
                                           weight=10.8,
                                           focus_idx_range=(0,0))
        
        ctrl_cost = rControl_act_ss(self.sys,
                                    state.pipeline_state,
                                    action,
                                    weight=-self._ctrl_cost_weight)
        
        torque_cost = rTorques(self.sys,
                               state.pipeline_state,
                               action,
                               weight=-0.0003)        

        upright_reward = rUpright(self.sys,
                                  state.pipeline_state,
                                  weight=0.001)
        
        healthy_reward = rHealthy_simple_z(self.sys,
                                           state.pipeline_state,
                                           self._healthy_z_range,
                                           early_terminate=self._terminate_when_unhealthy,
                                           weight=1.0,
                                           focus_idx_range=(0, 2))
        reward = 0.0
        reward = healthy_reward[0]
        reward += ctrl_cost 
        reward += torque_cost
        reward += upright_reward
        reward += waypoint_cost
        #reward += lin_vel_reward
        #reward += yaw_vel_reward
        
        obs = self._get_obs(pipeline_state, state.info)
        done = 1.0 - healthy_reward[1] if self._terminate_when_unhealthy else 0.0

        state.metrics.update(
            reward_ctrl = ctrl_cost,
            reward_alive = healthy_reward[0],
            reward_torque = torque_cost,
            reward_upright = upright_reward,
            #reward_lin_vel = lin_vel_reward,
            #reward_yaw_vel = yaw_vel_reward,
            reward_waypoint = waypoint_cost,
        )
        
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state, state_info) -> jax.Array:
        """Observe ant body position and velocities."""
        qpos = pipeline_state.q
        qvel = pipeline_state.qd
        #jcmd = state_info['jcmd']
        wcmd = state_info['wcmd']
        
        if self._exclude_current_positions_from_observation:
            qpos = pipeline_state.q[2:]

        return jp.concatenate([qpos] + [qvel] + [wcmd]) #[jcmd])

    def _sample_waypoint(self, rng: jax.Array) -> jax.Array:
        x_range = [-10, 10] 
        y_range = [-10, 10] 
        z_range = [0, 2]

        _, key1, key2, key3 = jax.random.split(rng, 4) 
        
        x = jax.random.uniform(
            key1, (1,), minval=x_range[0], maxval=x_range[1]        
        )
        
        y = jax.random.uniform(
            key1, (1,), minval=y_range[0], maxval=y_range[1]        
        )
        
        z = jax.random.uniform(
            key1, (1,), minval=z_range[0], maxval=z_range[1]        
        )

        wcmd = jp.array([x[0], y[0], z[0]])

        return wcmd
        
    def _sample_command(self, rng: jax.Array) -> jax.Array:
        lin_vel_x_range = [-0.6, 1.5]   #[m/s]
        lin_vel_y_range = [-0.6, 1.5]   #[m/s]
        yaw_vel_range = [-0.0, 0.0]     #[rad/s]

        _, key1, key2, key3 = jax.random.split(rng, 4)
        
        lin_vel_x = jax.random.uniform(
            key1, (1,), minval=lin_vel_x_range[0], maxval=lin_vel_x_range[1]        
        )

        lin_vel_y = jax.random.uniform(
            key2, (1,), minval=lin_vel_y_range[0], maxval=lin_vel_y_range[1]        
        )
        
        yaw_vel = jax.random.uniform(
            key3, (1,), minval=yaw_vel_range[0], maxval=yaw_vel_range[1]        
        )

        jcmd = jp.array([lin_vel_x[0], lin_vel_y[0], yaw_vel[0]])

        return jcmd

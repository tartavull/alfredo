from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp

from alfredo.tools import compose_scene
from alfredo.rewards import Reward
from alfredo.rewards import rConstant
from alfredo.rewards import rHealthy_simple_z
from alfredo.rewards import rControl_act_ss
from alfredo.rewards import rTorques
from alfredo.rewards import rTracking_lin_vel
from alfredo.rewards import rTracking_yaw_vel
from alfredo.rewards import rUpright
from alfredo.rewards import rTracking_Waypoint
from alfredo.rewards import rStand_still

class AAnt(PipelineEnv):
    """ """

    def __init__(self,
                 rewards = {},
                 env_xml_path = "",
                 agent_xml_path = "",
                 ctrl_cost_weight=0.5,
                 use_contact_forces=False,
                 contact_cost_weight=5e-4,
                 healthy_reward=1.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.3, 1.0),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 backend='generalized',
                 **kwargs,):

        # env_xml_path and agent_xml_path must be provided
        if env_xml_path and agent_xml_path:
            self._env_xml_path = env_xml_path
            self._agent_xml_path = agent_xml_path

            xml_scene = compose_scene(self._env_xml_path, self._agent_xml_path)
            sys = mjcf.loads(xml_scene)
        else:
            raise Exception("env_xml_path & agent_xml_path both must be provided") 

        # reward dictionary must be provided
        if rewards:
           self._rewards = rewards
        else:
           raise Exception("reward_Structure must be in kwargs")

        # TODO: clean this up in the future &
        #       make n_frames a function of input dt
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

        # Initialize the superclass "PipelineEnv"
        super().__init__(sys=sys, backend=backend, **kwargs)

        # Setting other object parameters based on input params
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

        # initialize position vector with minor randomization in pose
        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )

        # initialize velocity vector with minor randomization
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        # generate sample commands
        jcmd = self._sample_command(rng3)
        wcmd = jp.array([0.0, 0.0])

        # initialize pipeline_state (the physics state)
        pipeline_state = self.pipeline_init(q, qd)

        # reset values and metrics
        reward, done, zero = jp.zeros(3)

        state_info = {
            'jcmd':jcmd,
            'wcmd':wcmd,
            'rewards': {k: 0.0 for k in self._rewards.keys()},
            'step': 0,
        }

        metrics = {'pos_x_world_abs': zero,
                   'pos_y_world_abs': zero,
                   'pos_z_world_abs': zero,}

        for rn, r  in self._rewards.items():
            metrics[rn] = state_info['rewards'][rn]

        # get initial observation vector        
        obs = self._get_obs(pipeline_state, state_info)
       
        return State(pipeline_state, obs, reward, done, metrics, state_info)

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""

        # Save the previous physics state and step physics forward   
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        # Add all additional parameters to compute rewards
        self._rewards['r_lin_vel'].add_param('jcmd', state.info['jcmd'])
        self._rewards['r_yaw_vel'].add_param('jcmd', state.info['jcmd'])

        # Compute all rewards and accumulate total reward
        total_reward = 0.0
        for rn, r in self._rewards.items():
            r.add_param('sys', self.sys)
            r.add_param('pipeline_state', pipeline_state)

            reward_value = r.compute()
            state.info['rewards'][rn] = reward_value
            total_reward += reward_value[0]
            # print(f'{rn} reward_val = {reward_value}\n')

        # Computing additional metrics as necessary
        pos_world = pipeline_state.x.pos[0]
        abs_pos_world = jp.abs(pos_world)

        # Compute observations
        obs = self._get_obs(pipeline_state, state.info)
        done = 0.0

        # State management
        state.info['step'] += 1

        state.metrics.update(state.info['rewards'])
        
        state.metrics.update(
            pos_x_world_abs = abs_pos_world[0],
            pos_y_world_abs = abs_pos_world[1],
            pos_z_world_abs = abs_pos_world[2],
        )

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=total_reward, done=done
        )

    def _get_obs(self, pipeline_state, state_info) -> jax.Array:
        """Observe ant body position and velocities."""
        qpos = pipeline_state.q
        qvel = pipeline_state.qd

        inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
        local_rpyrate = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)
        torso_pos = pipeline_state.x.pos[0]
        
        jcmd = state_info['jcmd']
        #wcmd = state_info['wcmd']
        
        if self._exclude_current_positions_from_observation:
            qpos = pipeline_state.q[2:]

        obs = jp.concatenate([
            jp.array(qpos),
            jp.array(qvel),
            jp.array(local_rpyrate),
            jp.array(jcmd),            
        ])
        
        return obs 
    
    def _sample_waypoint(self, rng: jax.Array) -> jax.Array:
        x_range = [-25, 25]
        y_range = [-25, 25]
        z_range = [0, 2]

        _, key1, key2, key3 = jax.random.split(rng, 4) 
        
        x = jax.random.uniform(
            key1, (1,), minval=x_range[0], maxval=x_range[1]        
        )
        
        y = jax.random.uniform(
            key2, (1,), minval=y_range[0], maxval=y_range[1]        
        )
        
        z = jax.random.uniform(
            key3, (1,), minval=z_range[0], maxval=z_range[1]        
        )

        wcmd = jp.array([x[0], y[0]])

        return wcmd
        
    def _sample_command(self, rng: jax.Array) -> jax.Array:
        lin_vel_x_range = [-3.0, 3.0]   #[m/s]
        lin_vel_y_range = [-3.0, 3.0]   #[m/s]
        yaw_vel_range = [-1.0, 1.0]     #[rad/s]

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

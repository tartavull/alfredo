# pylint:disable=g-multiple-import
"""Trains Alfredo to run/walk/move in the +x direction."""
from typing import Tuple

import jax
from brax import actuator, base, math
from brax.envs import PipelineEnv, State
from brax.io import mjcf
from etils import epath
from jax import numpy as jp

from alfredo.tools import compose_scene
from alfredo.rewards import rConstant
from alfredo.rewards import rHealthy_simple_z
from alfredo.rewards import rSpeed_X
from alfredo.rewards import rControl_act_ss
from alfredo.rewards import rTorques

class Alfredo(PipelineEnv):
    # pyformat: disable
    """ """
    # pyformat: enable

    def __init__(
        self,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        backend="generalized",
        **kwargs,
    ):

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
        
        # this is vestigial - get rid of this someday soon
        if "scene_xml_path" in kwargs:
            path = kwargs["scene_xml_path"]
            del kwargs["scene_xml_path"]

            sys = mjcf.load(path)

        n_frames = 5

        if backend in ["spring", "positional"]:
            sys = sys.replace(dt=0.0015)
            n_frames = 10
            gear = jp.array(
                [
                    150.0,
                    150.0,
                    150.0,
                    150.0,
                    150.0,
                    150.0,
                    150.0,
                    150.0,
                    150.0,
                    150.0,
                    150.0,
                    150.0,
                    150.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                ]
            )  # pyformat: disable
            sys = sys.replace(actuator=sys.actuator.replace(gear=gear))

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward

        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale

        qpos = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(rng2, (self.sys.qd_size(),), minval=low, maxval=hi)

        pipeline_state = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))

        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_ctrl": zero,
            "reward_alive": zero,
            "reward_velocity": zero,
            "reward_torque":zero,
            "agent_x_position": zero,
            "agent_y_position": zero,
            "agent_x_velocity": zero,
            "agent_y_velocity": zero,
        }

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        prev_pipeline_state = state.pipeline_state
        pipeline_state = self.pipeline_step(prev_pipeline_state, action)
        obs = self._get_obs(pipeline_state, action)

        com_before, *_ = self._com(prev_pipeline_state)
        com_after, *_ = self._com(pipeline_state)

        x_speed_reward = rSpeed_X(self.sys,
                                  state.pipeline_state,
                                  CoM_prev=com_before,
                                  CoM_now=com_after,
                                  dt=self.dt,
                                  weight=self._forward_reward_weight)

        ctrl_cost = rControl_act_ss(self.sys,
                                    state.pipeline_state,
                                    action,
                                    weight=-self._ctrl_cost_weight)
        
        torque_cost = rTorques(self.sys,
                               state.pipeline_state,
                               action,
                               weight=-0.0003)        
        
        healthy_reward = rHealthy_simple_z(self.sys,
                                           state.pipeline_state,
                                           self._healthy_z_range,
                                           early_terminate=self._terminate_when_unhealthy,
                                           weight=self._healthy_reward,
                                           focus_idx_range=(0, 2))

        reward = healthy_reward[0] + ctrl_cost + x_speed_reward[0] + torque_cost

        done = 1.0 - healthy_reward[1] if self._terminate_when_unhealthy else 0.0

        state.metrics.update(
            reward_ctrl=ctrl_cost,
            reward_alive=healthy_reward[0],
            reward_velocity=x_speed_reward[0],
            reward_torque=torque_cost,
            agent_x_position=com_after[0],
            agent_y_position=com_after[1],
            agent_x_velocity=x_speed_reward[1],
            agent_y_velocity=x_speed_reward[2],
        )

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: base.State, action: jp.ndarray) -> jp.ndarray:
        """Observes Alfredo's body position, velocities, and angles."""

        a_positions = pipeline_state.q
        a_velocities = pipeline_state.qd
        #print(f"a_positions = {a_positions}")
        #print(f"a_velocities = {a_velocities}")

        if self._exclude_current_positions_from_observation:
            a_positions = a_positions[2:]

        com, inertia, mass_sum, x_i = self._com(pipeline_state)
        cinr = x_i.replace(pos=x_i.pos - com).vmap().do(inertia)
        com_inertia = jp.hstack(
            [cinr.i.reshape((cinr.i.shape[0], -1)), inertia.mass[:, None]]
        )

        xd_i = (
            base.Transform.create(pos=x_i.pos - pipeline_state.x.pos)
            .vmap()
            .do(pipeline_state.xd)
        )

        com_vel = inertia.mass[:, None] * xd_i.vel / mass_sum
        com_ang = xd_i.ang
        com_velocity = jp.hstack([com_vel, com_ang])

        qfrc_actuator = actuator.to_tau(
            self.sys, action, pipeline_state.q, pipeline_state.qd
        )

        # external_contact_forces are excluded
        return jp.concatenate(
            [
                a_positions,
                a_velocities,
                com_inertia.ravel(),
                com_velocity.ravel(),
                qfrc_actuator,
            ]
        )

    def _com(self, pipeline_state: base.State) -> jp.ndarray:
        """Computes Center of Mass of Alfredo"""

        inertia = self.sys.link.inertia

        if self.backend in ["spring", "positional"]:
            inertia = inertia.replace(
                i=jax.vmap(jp.diag)(
                    jax.vmap(jp.diagonal)(inertia.i)
                    ** (1 - self.sys.spring_inertia_scale)
                ),
                mass=inertia.mass ** (1 - self.sys.spring_mass_scale),
            )

        mass_sum = jp.sum(inertia.mass)
        x_i = pipeline_state.x.vmap().do(inertia.transform)

        com = jp.sum(jax.vmap(jp.multiply)(inertia.mass, x_i.pos), axis=0) / mass_sum

        return (
            com,
            inertia,
            mass_sum,
            x_i,
        )  # pytype: disable=bad-return-type  # jax-ndarray


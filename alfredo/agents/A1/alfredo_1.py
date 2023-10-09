# pylint:disable=g-multiple-import
"""Trains Alfredo to run/walk/move in the +x direction."""
from typing import Tuple

import jax
from brax import actuator, base, math
from brax.envs import PipelineEnv, State
from brax.io import mjcf
from etils import epath
from jax import numpy as jp


class Alfredo(PipelineEnv):
    # pyformat: disable
    """ """
    # pyformat: enable

    def __init__(
        self,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        backend="generalized",
        **kwargs,
    ):

        # forcing this model to need an input paramFile_path
        # will throw error if this is not included in kwargs
        path=""

        if "paramFile_path" in kwargs:
            path = kwargs["paramFile_path"]
            del kwargs["paramFile_path"]

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
        a_velocity = (com_after - com_before) / self.dt

        reward_vel = math.safe_norm(a_velocity)
        forward_reward = self._forward_reward_weight * a_velocity[0]  # * reward_vel
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, x=0.0, y=1.0)
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] > max_z, x=0.0, y=is_healthy)

        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        reward = healthy_reward - ctrl_cost + forward_reward

        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        state.metrics.update(
            reward_ctrl=-ctrl_cost,
            reward_alive=healthy_reward,
            reward_velocity=forward_reward,
            agent_x_position=com_after[0],
            agent_y_position=com_after[1],
            agent_x_velocity=a_velocity[0],
            agent_y_velocity=a_velocity[1],
        )

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: base.State, action: jp.ndarray) -> jp.ndarray:
        """Observes humanoid body position, velocities, and angles."""

        a_positions = pipeline_state.q
        a_velocities = pipeline_state.qd

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
        """Computes Center of Mass of the Humanoid"""

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


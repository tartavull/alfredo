import pytest


def test_gpu_availability():
    import torch


def test_gym():
    import gym

    env = gym.make("MountainCar-v0")
    observation, info = env.reset(seed=42)
    action = env.action_space.sample()
    env.close()
    assert observation[0] == pytest.approx(-0.4452088, 0.001)

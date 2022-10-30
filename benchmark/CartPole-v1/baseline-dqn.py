import gym
from tqdm.rich import trange
from random import choice, randint, random
from stable_baselines3 import DQN
import time

def render(model=None):
    env = gym.make("CartPole-v1")
    obs = env.reset()
    while True:
        if model:
            action, _states = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            time.sleep(1)
            obs = env.reset()
    env.close()

def train():
    # hyperparamets from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml
    env = gym.make("CartPole-v1")
    model = DQN(
        policy="MlpPolicy", 
        policy_kwargs=dict(net_arch=[256, 256]),
        env=env, 
        learning_rate=2.3e-3,
        batch_size=64,
        buffer_size=100000,
        learning_starts=1000,
        gamma=0.99,
        target_update_interval=10,
        train_freq=256,
        gradient_steps=128,
        exploration_fraction=0.16,
        exploration_final_eps=0.04,
        verbose=1,
    )
    model.learn(
        total_timesteps=5e4, 
        log_interval=4,
        progress_bar=True,
    )
    return model

if __name__ == "__main__":
    render(train())

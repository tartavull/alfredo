import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import time

def render(model=None):
    env = gym.make("BipedalWalker-v3")
    obs = env.reset()
    while True:
        if model:
            action, _states = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        obs, _reward, done, _info = env.step(action)
        env.render()
        if done:
            time.sleep(1)
            obs = env.reset()
    env.close()

def train():
    # hyperparameters from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/a2c.yml
    env = make_vec_env("BipedalWalker-v3", n_envs=16)

    model = A2C(
        policy="MlpPolicy", 
        env=env, 
        policy_kwargs=dict(log_std_init=-2, ortho_init=False),
        ent_coef=0.0,
        max_grad_norm=0.5,
        n_steps=8,
        gae_lambda=0.9,
        vf_coef=0.4,
        gamma=0.99,
        use_rms_prop=True,
        normalize_advantage=False,
        learning_rate=0.00096,
        use_sde=True,
        verbose=1)

    model.learn(
        total_timesteps=5e6, 
        progress_bar=True,
        log_interval=4)
    return model

if __name__ == "__main__":
    render(train())

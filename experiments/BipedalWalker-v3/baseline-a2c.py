import gym
from genetic_intelligence import never_ending_render, train_model
from stable_baselines3 import A2C


def train():
    # hyperparameters from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/a2c.yml
    config = {
        "policy": "MlpPolicy",
        "total_timesteps": 5e6,
        "env_name": "BipedalWalker-v3",
        "n_envs": 16,
        "policy_kwargs": dict(log_std_init=-2, ortho_init=False),
        "ent_coef": 0.0,
        "max_grad_norm": 0.5,
        "n_steps": 8,
        "gae_lambda": 0.9,
        "vf_coef": 0.4,
        "gamma": 0.99,
        "use_rms_prop": True,
        "normalize_advantage": False,
        "learning_rate": 0.00096,
        "use_sde": True,
    }
    return train_model(config, A2C)


if __name__ == "__main__":
    never_ending_render("BipedalWalker-v3", train())

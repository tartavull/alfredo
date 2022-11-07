from stable_baselines3 import DQN
from genetic_intelligence import never_ending_render, train_model

def train():
    # hyperparamets from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml
    config = {
        "env_name": "CartPole-v1",
        "n_envs": 4,
        "total_timesteps": 5e4,
        "policy": "MlpPolicy",
        "policy_kwargs": dict(net_arch=[256, 256]),
        "learning_rate": 2.3e-3,
        "batch_size":64,
        "buffer_size":100000,
        "learning_starts":1000,
        "gamma":0.99,
        "target_update_interval":10,
        "train_freq":256,
        "gradient_steps":128,
        "exploration_fraction":0.16,
        "exploration_final_eps":0.04,
    }
    return train_model(config, DQN)


if __name__ == "__main__":
    never_ending_render("CartPole-v1", train())

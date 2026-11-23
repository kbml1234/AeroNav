from stable_baselines3 import DQN

# the plan's starting settings, not tuned
DQN_SETTINGS = {
    "learning_rate": 3e-4,
    "buffer_size": 50_000,
    "learning_starts": 2_000,
    "batch_size": 64,
    "gamma": 0.99,
    "exploration_fraction": 0.3,
    "exploration_final_eps": 0.05,
    "policy_kwargs": {"net_arch": [64, 64]},
}


def make_dqn(env, seed=0, **changes):
    settings = {**DQN_SETTINGS, **changes}
    return DQN("MlpPolicy", env, seed=seed, verbose=0, **settings)

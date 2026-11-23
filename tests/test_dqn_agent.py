import numpy as np
import torch
from stable_baselines3 import DQN

from src.dqn_agent import make_dqn
from src.drone_env import DroneNavEnv


def weights(model):
    return [p.detach().clone() for p in model.q_net.parameters()]


def test_network_shape_is_8_64_64_4():
    model = make_dqn(DroneNavEnv())
    layers = [m for m in model.q_net.q_net if isinstance(m, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in layers] == [(8, 64), (64, 64), (64, 4)]


def test_smoke_100_steps_runs_and_learns_something():
    model = make_dqn(DroneNavEnv(), learning_starts=50)   # so learning steps start inside the 100
    before = weights(model)
    model.learn(total_timesteps=100)
    after = weights(model)
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_predict_gives_a_valid_action():
    env = DroneNavEnv()
    model = make_dqn(env)
    obs, info = env.reset(options={"map_seed": 42})
    action, _ = model.predict(obs, deterministic=True)
    assert int(action) in {0, 1, 2, 3}


def test_save_and_load_give_the_same_answers(tmp_path):
    env = DroneNavEnv()
    model = make_dqn(env, learning_starts=50)
    model.learn(total_timesteps=100)
    model.save(tmp_path / "dqn")
    loaded = DQN.load(tmp_path / "dqn")
    for seed in range(10):
        obs, info = env.reset(options={"map_seed": seed})
        a1, _ = model.predict(obs, deterministic=True)
        a2, _ = loaded.predict(obs, deterministic=True)
        assert a1 == a2
    obs = np.array([obs])
    assert torch.equal(model.q_net(torch.as_tensor(obs)), loaded.q_net(torch.as_tensor(obs)))


def test_frozen_final_model_matches_its_model_card():
    import hashlib
    import json
    from pathlib import Path
    card = json.loads(Path("results/models/final/model_card.json").read_text())
    model_file = Path(card["file"])
    assert hashlib.sha256(model_file.read_bytes()).hexdigest() == card["sha256"]
    model = DQN.load(model_file)
    assert model.observation_space.shape == (12,)
    assert sum(p.numel() for p in model.policy.q_net.parameters()) == card["training"]["parameters"] == 5252


def test_frozen_final_model_flies_validation_map_20000_to_the_goal_in_14_moves():
    from src.evaluate import fly_dqn
    model = DQN.load("results/models/final/dqn_calm_last_move_500k.zip")
    result = fly_dqn(DroneNavEnv(include_last_move=True), model, 20000)
    assert result["ending"] == "goal" and result["steps"] == 14

import numpy as np

N_ACTIONS = 4


def make_q_table(size):
    return np.zeros((size, size, N_ACTIONS))


def best_action(q_table, pos, rng):
    # random tie-break (argmax would always pick the first)
    scores = q_table[pos]
    tied = np.flatnonzero(scores == scores.max())
    return int(rng.choice(tied))


def choose_action(q_table, pos, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))
    return best_action(q_table, pos, rng)


def q_update(q_table, pos, action, reward, next_pos, terminated, alpha, gamma):
    # goal or crash: no future. A timeout keeps its future (the drone could have kept flying)
    future = 0.0 if terminated else q_table[next_pos].max()
    target = reward + gamma * future
    old = q_table[pos][action]
    q_table[pos][action] = old + alpha * (target - old)   # changes the caller's table (arrays are shared)
    return q_table[pos][action]


def epsilon_at(episode, episodes, eps_start, eps_end, decay_fraction=0.8):
    decay_episodes = decay_fraction * episodes
    if episode >= decay_episodes:
        return eps_end
    return eps_start + (eps_end - eps_start) * episode / decay_episodes


def train(env, map_seed, episodes, alpha, gamma, eps_start=1.0, eps_end=0.05, seed=0):
    rng = np.random.default_rng(seed)
    q_table = make_q_table(env.size)
    history = []
    for episode in range(episodes):
        epsilon = epsilon_at(episode, episodes, eps_start, eps_end)
        env.reset(options={"map_seed": map_seed})
        total_reward = 0.0
        while True:
            pos = env.pos
            action = choose_action(q_table, pos, epsilon, rng)
            obs, reward, terminated, truncated, info = env.step(action)
            q_update(q_table, pos, action, reward, env.pos, terminated, alpha, gamma)
            total_reward += reward
            if terminated or truncated:
                break
        ending = info["event"] if terminated else "timeout"
        history.append({"episode": episode, "reward": total_reward, "steps": env.steps,
                        "ending": ending, "epsilon": epsilon})
    return q_table, history


def run_greedy(env, q_table, map_seed, seed=0):
    rng = np.random.default_rng(seed)
    env.reset(options={"map_seed": map_seed})
    path = [env.pos]
    while True:
        obs, reward, terminated, truncated, info = env.step(choose_action(q_table, env.pos, 0.0, rng))
        if env.pos != path[-1]:
            path.append(env.pos)
        if terminated or truncated:
            ending = info["event"] if terminated else "timeout"
            return {"ending": ending, "steps": env.steps, "path": path}

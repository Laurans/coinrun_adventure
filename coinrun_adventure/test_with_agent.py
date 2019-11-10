from coinrun_adventure.utils import setup
from coinrun import make
import numpy as np


def test(model):
    def run(seed):
        setup(
            rand_seed=seed,
            num_envs=1,
            high_difficulty=False,
            num_levels=0,
            use_data_augmentation=False,
        )
        env = make("standard", num_envs=1)
        obs = env.reset()
        episode_rew = 0
        done = False
        while not done:
            actions, _, _ = model.get_all_values(obs)
            actions = actions.numpy()
            next_obs, rew, done, _, = env.step(actions)
            obs = next_obs
            done = done.any() if isinstance(done, np.ndarray) else done
            episode_rew += rew

        return episode_rew

    mean_rewards = [0, 0]
    start = 3000
    end = start + 500
    for seed in range(start, end):
        rew = run(seed)
        mean_rewards[0] += 1
        mean_rewards[1] += 1 if rew > 0 else 0
        # print(seed % start, "Reward received", rew)
    print("Mean ", mean_rewards[1] / mean_rewards[0])

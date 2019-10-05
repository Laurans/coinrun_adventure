import numpy as np
from coinrun import make
from coinrun_adventure import setup_utils


def random_agent(num_envs=3, max_steps=100000):
    setup_utils.setup()
    env = make("standard", num_envs=num_envs)
    env.reset()
    for step in range(max_steps):
        acts = np.array([env.action_space.sample() for _ in range(env.num_envs)])
        _obs, rews, _dones, _infos = env.step(acts)
        breakpoint()
        print("step", step, "rews", rews)
    env.close()


if __name__ == "__main__":
    random_agent()

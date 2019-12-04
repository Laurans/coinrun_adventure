from collections import defaultdict

import numpy as np
import torch

from coinrun_adventure.utils.torch_utils import input_preprocessing, tensor, to_np


class Runner:
    def __init__(self, env, model, num_steps, gamma_coef, lambda_coef, device):
        self.env = env
        self.model = model
        self.num_steps = num_steps
        self.lam = lambda_coef
        self.gamma = gamma_coef
        self.device = device

        self.obs = np.zeros(
            (env.num_envs,) + env.observation_space.shape,
            dtype=env.observation_space.dtype.name,
        )

        self.obs[:] = env.reset()
        self.dones = [False for _ in range(env.num_envs)]

    def run(self):
        storage = defaultdict(list)
        epinfos = []
        self.model.eval()

        with torch.no_grad():

            for _ in range(self.num_steps):
                obs = input_preprocessing(self.obs, device=self.device)
                prediction = self.model.step(obs)
                actions = to_np(prediction["action"])
                self.obs[:], rewards, self.dones, infos = self.env.step(actions)

                storage["states"] += [obs]
                storage["actions"] += [prediction["action"]]
                storage["values"] += [prediction["state_value"]]
                storage["log_prob_a"] += [prediction["log_prob_a"]]
                storage["entropy"] += [prediction["entropy"]]
                storage["rewards"] += [tensor(rewards, device=self.device)]
                storage["masks"] += [tensor(1 - self.dones, device=self.device)]
                storage["advantages"].append([None] * self.env.num_envs)
                storage["returns"].append([None] * self.env.num_envs)
                for info in infos:
                    maybeepinfo = info.get("episode")
                    if maybeepinfo:
                        epinfos.append(maybeepinfo)

            lastvalues = self.model.step(
                input_preprocessing(self.obs, device=self.device)
            )["state_value"]
            returns = lastvalues

            advantages = tensor(np.zeros((self.env.num_envs, 1)), device=self.device)
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextvalues = lastvalues
                else:
                    nextvalues = storage["values"][t + 1]

                returns = (
                    storage["rewards"][t] + self.gamma * storage["masks"][t] * returns
                )

                td_error = (
                    storage["rewards"][t]
                    + self.gamma * storage["masks"][t] * nextvalues
                    - storage["values"][t]
                )

                advantages = (
                    advantages * self.lam * self.gamma * storage["masks"][t] + td_error
                )

                storage["advantages"][t] = advantages
                storage["returns"][t] = returns

        for key in list(storage.keys()):
            storage[key] = torch.cat(storage[key], dim=0)

        return storage, epinfos

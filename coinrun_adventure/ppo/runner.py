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
        self.dones = np.array([False for _ in range(env.num_envs)])

    def run(self):
        storage = defaultdict(list)
        epinfos = []
        self.model.eval()

        with torch.no_grad():

            for _ in range(self.num_steps):
                obs = input_preprocessing(self.obs, device=self.device)
                prediction = self.model.step(obs)
                actions = to_np(prediction["action"])
                storage["states"] += [to_np(obs.clone())]
                storage["actions"] += [to_np(prediction["action"])]
                storage["values"] += [to_np(prediction["state_value"].squeeze())]
                storage["log_prob_a"] += [to_np(prediction["log_prob_a"].squeeze())]
                storage["dones"] += [self.dones]

                self.obs[:], rewards, self.dones, infos = self.env.step(actions)
                storage["rewards"] += [rewards]
                for info in infos:
                    maybeepinfo = info.get("episode")
                    if maybeepinfo:
                        epinfos.append(maybeepinfo)


            # batch of steps to batch of rollouts
            for key in storage:
                storage[key] = np.asarray(storage[key])

            lastvalues = to_np(self.model.step(
                input_preprocessing(self.obs, device=self.device)
            )["state_value"].squeeze())

            # discount/bootstrap
            storage['advantages'] = np.zeros_like(storage['rewards'])
            storage['returns'] = np.zeros_like(storage['rewards'])

            lastgaelam = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - self.dones
                    nextvalues = lastvalues
                else:
                    nextnonterminal = 1.0 - storage['dones'][t+1]
                    nextvalues = storage["values"][t + 1]

                td_error = (
                    storage["rewards"][t]
                    + self.gamma * nextvalues * nextnonterminal
                    - storage["values"][t]
                )

                storage['advantages'][t] = lastgaelam = (td_error + self.gamma *self.lam * nextnonterminal * lastgaelam
                )

            storage["returns"] = storage['advantages'] + storage['values']

        for key in storage:
            s = storage[key].shape
            storage[key] = storage[key].swapaxes(0,1).reshape(s[0]*s[1], *s[2:])

        return storage, epinfos

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
                storage["states"] += [obs.clone()]
                storage["actions"] += [prediction["action"]]
                storage["values"] += [prediction["state_value"].squeeze()]
                storage["log_prob_a"] += [prediction["log_prob_a"].squeeze()]
                storage["dones"] += [tensor(self.dones, device=self.device)]

                self.obs[:], rewards, self.dones, infos = self.env.step(actions)
                storage["rewards"] += [tensor(rewards, device=self.device)]
                for info in infos:
                    maybeepinfo = info.get("episode")
                    if maybeepinfo:
                        epinfos.append(maybeepinfo)


            # batch of steps to batch of rollouts
            for key in storage:
                storage[key] = torch.stack(storage[key])

            lastvalues = self.model.step(
                input_preprocessing(self.obs, device=self.device)
            )["state_value"]

            # discount/bootstrap
            storage['advantages'] = torch.zeros_like(storage['rewards'])
            storage['returns'] = torch.zeros_like(storage['rewards'])

            lastgaelam = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = tensor(1.0 - self.dones, device=self.device)
                    nextvalues = lastvalues.squeeze()
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
            storage[key] = storage[key].transpose(0,1).reshape(s[0]*s[1], *s[2:])

        return storage, epinfos

import tensorflow as tf
import numpy as np


class PPOAgent:
    def __init__(self, env, model, num_steps, gamma_coef, lambda_coef):
        self.env = env
        self.model = model
        self.batch_ob_shape = ()
        self.lam = lambda_coef
        self.gamma = gamma_coef
        self.num_steps = num_steps

        self.obs = np.zeros(
            (env.num_envs,) + env.observation_shape.shape,
            dtype=env.observation_shape.dtype.name,
        )
        self.obs[:] = env.reset()
        self.dones = [False for _ in range(env.num_envs)]

    def step(self):
        """
        Make a mini batch
        """
        # Here, we init the lists that will contain the mb of experiences
        # NOTE: mb?
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        # For n in range number of steps
        for _ in range(self.num_steps):
            # Given observations, get action value and neglogpacs
            # We already have self.obs
            obs = tf.constant(self.obs)
            actions, values, self.states, neglogpacs = self.model.step(obs)
            actions = actions._numpy()

            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values._numpy())
            mb_neglogpacs.append(neglogpacs._numpy())
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            mb_rewards.append(rewards)

        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(tf.constant(self.obs))._numpy()

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t + 1]
                nextvalues = mb_values[t + 1]

            delta = (
                mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            )
            mb_advs[t] = lastgaelam = (
                delta + self.gamma * self.lam * nextnonterminal * lastgaelam
            )

        mb_returns = mb_advs + mb_values
        return map(
            sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)
        )


def sf01(arr):
    """
    Swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

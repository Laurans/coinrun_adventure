import tensorflow as tf
from .networks import fc
from gym import spaces
from .distributions import DiagGaussianPdType, CategoricalPdType


def make_pdtype(latent_shape, ac_space, init_scale=1.0):
    # TODO: Replace by tensorflow-probability

    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        return DiagGaussianPdType(latent_shape, ac_space.shape[0], init_scale)
    elif isinstance(ac_space, spaces.Discrete):
        return CategoricalPdType(latent_shape, ac_space.n, init_scale)
    else:
        raise ValueError("No implementation for {}".format(ac_space))


class PolicyWithValue(tf.Module):
    def __init__(self, ac_space, policy_network, value_network=None, estimate_q=False):
        """
        ac_space    action space
        policy_network  keras network for policy
        value_network   keras network for value
        estimate_q      q_value or v value

        """
        self.policy_network = policy_network
        self.value_network = value_network or policy_network
        self.estimate_q = estimate_q
        self.initial_state = None

        # Based on the action space, will select what probability distribution type
        self.pdtype = make_pdtype(
            policy_network.output_shape, ac_space, init_scale=0.01
        )

        if estimate_q:
            assert isinstance(ac_space, spaces.Discrete)
            # self.pi = fc("pi", self.policy_network.output_shape, ac_space.n)
            self.value_fc = fc(
                "q", input_shape=self.value_network.output_shape, units=ac_space.n
            )
        else:
            # self.pi = fc("pi", self.policy_network.output_shape, ac_space.shape[0])
            self.value_fc = fc(
                "vf", input_shape=self.value_network.output_shape, units=1
            )

    @tf.function
    def step(self, observation):
        latent = self.policy_network(observation)
        pd, pi = self.pdtype.pdfromlatent(latent)
        action = pd.sample()
        neglogp = pd.neglogp(action)
        value_latent = self.value_network(observation)
        vf = tf.squeeze(self.value_fc(value_latent), axis=1)
        return action, vf, None, neglogp

    @tf.function
    def value(self, observation):
        value_latent = self.value_network(observation)
        result = tf.squeeze(self.value_fc(value_latent), axis=1)
        return result

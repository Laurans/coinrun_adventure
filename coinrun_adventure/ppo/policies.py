import tensorflow as tf
from coinrun_adventure.common.networks import get_network_builder, fc
from gym import spaces
from coinrun_adventure.common import CategoricalPd


class Policy(tf.Module):
    def __init__(
        self, policy_network_architecture, ob_shape, ac_space, estimate_q=False
    ):
        super(Policy, self).__init__()
        original_inputs = tf.keras.Input(shape=ob_shape, dtype=tf.uint8)

        self.policy_network = get_network_builder(policy_network_architecture)(
            original_inputs
        )
        self.value_network = self.policy_network

        self.initial_state = None

        if estimate_q:
            assert isinstance(ac_space, spaces.Discrete)
            # self.pi = fc("pi", self.policy_network.output_shape, ac_space.n)
            self.value_fc = get_network_builder("fc")(
                "q", self.value_network, units=ac_space.n
            )
        else:
            # self.pi = fc("pi", self.policy_network.output_shape, ac_space.shape[0])
            value_fc = get_network_builder("fc")(
                "vf", x_input=self.value_network, units=1
            )
            self.value_fc = tf.keras.Model(inputs=original_inputs, outputs=value_fc)

        pi = get_network_builder("fc")(
            "pi",
            x_input=self.policy_network,
            units=ac_space.n,
            init_scale=0.01,
            init_bias=0.0,
        )
        self.pi = tf.keras.Model(inputs=original_inputs, outputs=pi)

        # Based on the action space, will select what probability distribution type
        self.distribution = CategoricalPd

    def raw_value(self, observation):
        out_pi = self.pi(observation)
        distribution = self.distribution(out_pi)
        action = distribution.sample()
        vf = self.value(observation)
        return action, vf, out_pi

    @tf.function
    def step(self, observation):
        out_pi = self.pi(observation)
        distribution = self.distribution(out_pi)
        action = distribution.sample()
        neglogp = distribution.neglogp(action)
        vf = self.value(observation)
        return action, vf, None, neglogp

    @tf.function
    def value(self, observation):
        result = tf.squeeze(self.value_fc(observation), axis=1)
        return result


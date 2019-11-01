import tensorflow as tf
from .policies import Policy
from mpi4py import MPI
from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
from baselines.common.mpi_util import sync_from_root


class Model(tf.Module):
    def __init__(
        self,
        ob_shape,
        ac_space,
        policy_network_archi,
        ent_coef,
        vf_coef,
        l2_coef,
        max_grad_norm,
        sync_from_root_value,
    ):
        super().__init__(name="PPO2Model")
        self.network = Policy(policy_network_archi, ob_shape, ac_space)
        if sync_from_root_value:
            self.optimizer = MpiAdamOptimizer(
                MPI.COMM_WORLD, self.network.trainable_variables
            )
        else:
            self.optimizer = tf.keras.optimizers.Adam()
        self.optimizer.epsilon = 1e-5
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.l2_coef = l2_coef
        self.max_grad_norm = max_grad_norm
        self.step = self.network.step
        self.value = self.network.value
        self.get_all_values = self.network.raw_value
        self.initial_state = self.network.initial_state
        self.loss_names = [
            "policy_loss",
            "value_loss",
            "policy_entropy",
            "approxkl",
            "clipfrac",
        ]
        if sync_from_root_value:
            sync_from_root(self.variables)

        self.sync_from_root_value = sync_from_root_value

    def train(self, lr, cliprange, obs, returns, masks, actions, values, neglogpac_old):
        """
        Make the training part (feedforward and retropropagation of gradients)
        """
        grads, pg_loss, vf_loss, entropy, approxkl, clipfrac = self.get_grad(
            cliprange, obs, returns, masks, actions, values, neglogpac_old
        )

        if self.sync_from_root_value:
            self.optimizer.apply_gradients(grads, lr)
        else:
            self.optimizer.learning_rate = lr
            grads_and_vars = zip(grads, self.network.trainable_variables)
            self.optimizer.apply_gradients(grads_and_vars)

        return pg_loss, vf_loss, entropy, approxkl, clipfrac

    def get_grad(self, cliprange, obs, returns, masks, actions, values, neglogpac_old):
        # we calcurate advantage A(s, a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - tf.reduce_mean(advs)) / (tf.keras.backend.std(advs) + 1e-8)

        var_list = self.network.trainable_variables
        weight_params = [v for v in var_list if "/b" not in v.name]

        with tf.GradientTape() as tape:
            out_pi = self.network.pi(obs)
            distribution = self.network.distribution(out_pi)
            neglogpac = distribution.neglogp(actions)
            entropy = tf.reduce_mean(distribution.entropy())
            vpred = self.network.value(obs)
            vpredclipped = values + tf.clip_by_value(
                vpred - values, -cliprange, cliprange
            )
            vf_losses1 = tf.square(vpred - returns)
            vf_losses2 = tf.square(vpredclipped - returns)
            vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

            ratio = tf.exp(neglogpac_old - neglogpac)
            pg_losses1 = -advs * ratio
            pg_losses2 = -advs * tf.clip_by_value(ratio, 1 - cliprange, 1 + cliprange)
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses1, pg_losses2))

            approxkl = 0.5 * tf.reduce_mean(tf.square(neglogpac - neglogpac_old))
            clipfrac = tf.reduce_mean(
                tf.cast(tf.greater(tf.abs(ratio - 1.0), cliprange), tf.float32)
            )

            l2_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in weight_params])

            loss = (
                pg_loss
                - entropy * self.ent_coef
                + vf_loss * self.vf_coef
                + l2_loss
                + self.l2_coef
            )

        grads = tape.gradient(loss, var_list)

        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)

        if self.sync_from_root_value:
            grads = tf.concat([tf.reshape(g, (-1,)) for g in grads], axis=0)

        return grads, pg_loss, vf_loss, entropy, approxkl, clipfrac

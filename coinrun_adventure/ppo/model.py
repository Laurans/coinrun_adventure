import tensorflow as tf
from coinrun_adventure.common.policies import PolicyWithValue


class Model(tf.Module):
    def __init__(
        self, ac_space, policy_network, value_network, ent_coef, vf_coef, max_grad_norm
    ):
        super().__init__(name="PPO2Model")
        self.train_model = PolicyWithValue(
            ac_space, policy_network, value_network, estimate_q=False
        )
        self.optimizer = tf.keras.optimizers.Adam()
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.step = self.train_model.step
        self.value = self.train_model.value
        self.initial_state = self.train_model.initial_state
        self.loss_names = [
            "policy_loss",
            "value_loss",
            "policy_entropy",
            "approxkl",
            "clipfrac",
        ]

    def train(self, lr, cliprange, obs, returns, masks, actions, values, neglogpac_old):
        """
        Make the training part (feedforward and retropropagation of gradients)
        """
        grads, pg_loss, vf_loss, entropy, approxkl, clipfrac = self.get_grad(
            cliprange, obs, returns, masks, actions, values, neglogpac_old
        )

        self.optimizer.learning_rate = lr
        grads_and_vars = zip(grads, self.train_model.trainable_variables)
        self.optimizer.apply_gradients(grads_and_vars)

        return pg_loss, vf_loss, entropy, approxkl, clipfrac

    @tf.function
    def get_grad(self, cliprange, obs, returns, masks, actions, values, neglogpac_old):
        # we calcurate advantage A(s, a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - tf.reduce_mean(advs)) / (tf.keras.backend.std(advs) + 1e-8)

        with tf.GradientTape() as tape:
            policy_latent = self.train_model.policy_network(obs)
            pd, _ = self.train_model.pdtype.pdfromlatent(policy_latent)
            neglogpac = pd.neglogp(actions)
            entropy = tf.reduce_mean(pd.entropy())
            vpred = self.train_model.value(obs)
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

            loss = pg_loss - entropy * self.ent_coef + vf_loss * self.vf_coef

        var_list = self.train_model.trainable_variables
        grads = tape.gradient(loss, var_list)

        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        return grads, pg_loss, vf_loss, entropy, approxkl, clipfrac


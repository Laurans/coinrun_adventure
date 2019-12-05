from coinrun_adventure.network.bodies import body_factory
from coinrun_adventure.network.heads import CategoricalActorCriticPolicy
import torch
from coinrun_adventure.utils.torch_utils import sync_gradients


class Model:
    def __init__(
        self,
        ob_shape: tuple,
        ac_space: int,
        policy_network_archi,
        ent_coef,
        vf_coef,
        l2_coef,
        max_grad_norm,
        device,
    ):
        phi_body = body_factory(policy_network_archi)(CHW_shape=ob_shape[::-1])
        actor_body = body_factory("DummyBody")(phi_body.feature_dim)
        critic_body = body_factory("DummyBody")(phi_body.feature_dim)

        self.network = CategoricalActorCriticPolicy(
            CHW_shape=ob_shape,
            action_dim=ac_space,
            phi_body=phi_body,
            actor_body=actor_body,
            critic_body=critic_body,
        )
        self.network.to(device)
        self.device = device

        self.optimizer = torch.optim.Adam(self.network.parameters(), eps=1e-5)
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.l2_coef = l2_coef
        self.max_grad_norm = max_grad_norm
        self.step = self.network.forward
        self.loss_names = [
            "policy_loss",
            "value_loss",
            "policy_entropy",
            "approxkl",
            "clipfrac",
        ]

    def eval(self):
        self.network.eval()

    def train(self, lr: float, cliprange, batch: dict):
        self.network.train()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        states = batch["states"]
        actions = batch["actions"]
        log_probs_old = batch["log_prob_a"]
        returns = batch["returns"]
        values = batch["values"]

        advantages = returns - values
        advantages = (advantages - advantages.mean())/(advantages.std() + 1e-8)

        prediction = self.network(obs=states, action=actions)

        policy_entropy = prediction["entropy"].mean()

        vpredclipped = values + (prediction["state_value"] - values).clamp(
            -cliprange, cliprange
        )
        value_loss1 = (returns - prediction["state_value"]).pow(2)
        value_loss2 = (returns - vpredclipped).pow(2)
        value_loss = 0.5 * (torch.max(value_loss1, value_loss2)).mean()

        ratio = (prediction["log_prob_a"] - log_probs_old).exp()
        policy_loss1 = -advantages * ratio
        policy_loss2 = -advantages * ratio.clamp(1.0 - cliprange, 1.0 + cliprange)
        policy_loss = (torch.max(policy_loss1, policy_loss2)).mean()

        approxkl = 0.5 * ((prediction["log_prob_a"] - log_probs_old).pow(2)).mean()
        clipfrac = (torch.abs(ratio - 1.0) > cliprange).float().mean()

        l2_reg = torch.tensor(0.0, device=self.device)
        for param in self.network.parameters():
            l2_reg += torch.norm(param)

        loss = (
            policy_loss
            - policy_entropy * self.ent_coef
            + value_loss * self.vf_coef
            + l2_reg * self.l2_coef
        )

        self.optimizer.zero_grad()
        loss.backward()
        sync_gradients(self.network)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return policy_loss, value_loss, policy_entropy, approxkl, clipfrac

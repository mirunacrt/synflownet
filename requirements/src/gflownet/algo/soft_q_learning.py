import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data as gd
from torch import Tensor
from torch_scatter import scatter

from gflownet.algo.graph_sampling import GraphSampler, Sampler
from gflownet.config import Config
from gflownet.envs.graph_building_env import GraphBuildingEnv, GraphBuildingEnvContext, generate_forward_trajectory
from gflownet.utils.misc import get_worker_device


class SoftQLearning:
    def __init__(
        self,
        env: GraphBuildingEnv,
        ctx: GraphBuildingEnvContext,
        cfg: Config,
        sampler: Sampler,
    ):
        """Soft Q-Learning implementation, see
        Haarnoja, Tuomas, Haoran Tang, Pieter Abbeel, and Sergey Levine. "Reinforcement learning with deep
        energy-based policies." In International conference on machine learning, pp. 1352-1361. PMLR, 2017.

        Hyperparameters used:
        illegal_action_logreward: float, log(R) given to the model for non-sane end states or illegal actions

        Parameters
        ----------
        env: GraphBuildingEnv
            A graph environment.
        ctx: GraphBuildingEnvContext
            A context.
        cfg: Config
            The experiment configuration
        """
        self.ctx = ctx
        self.env = env
        self.cfg = cfg
        self.max_len = cfg.algo.max_len
        self.max_nodes = cfg.algo.max_nodes
        self.illegal_action_logreward = cfg.algo.illegal_action_logreward
        self.alpha = cfg.algo.sql.alpha
        self.gamma = cfg.algo.sql.gamma
        self.invalid_penalty = cfg.algo.sql.penalty
        self.bootstrap_own_reward = False
        # Experimental flags
        self.sample_temp = 1
        self.do_q_prime_correction = cfg.algo.sql.do_q_prime_correction
        # self.graph_sampler = GraphSampler(ctx, env, self.max_len, self.max_nodes, self.sample_temp)
        self.graph_sampler = sampler
        self.updates = 0
        self.lagged_model_update_freq = cfg.algo.sql.lagged_model_update_freq
        self.lagged_model_tau = cfg.algo.sql.lagged_model_tau
        self.lagged_model = None
        self.random_action_prob = {
            "train": cfg.algo.train_random_action_prob,
            "eval": cfg.algo.valid_random_action_prob,
        }
        self.is_eval = True

    def set_is_eval(self, is_eval: bool):
        self.is_eval = is_eval

    def step(self):
        self.updates += 1  # This isn't used anywhere?

    def create_training_data_from_own_samples(
        self, model: nn.Module, n: int, cond_info: Tensor, random_action_prob: float
    ):
        """Generate trajectories by sampling a model

        Parameters
        ----------
        model: nn.Module
           The model being sampled
        graphs: List[Graph]
            List of N Graph endpoints
        cond_info: torch.tensor
            Conditional information, shape (N, n_info)
        random_action_prob: float
            Probability of taking a random action
        Returns
        -------
        data: List[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: List[Tuple[Graph, GraphAction]]
           - fwd_logprob: log Z + sum logprobs P_F
           - bck_logprob: sum logprobs P_B
           - is_valid: is the generated graph valid according to the env & ctx
        """
        dev = get_worker_device()
        cond_info = cond_info.to(dev)
        data = self.graph_sampler.sample_from_model(
            model, n, cond_info, random_action_prob, use_argmax=self.alpha == 0.0
        )
        return data

    def create_training_data_from_graphs(self, graphs):
        """Generate trajectories from known endpoints

        Parameters
        ----------
        graphs: List[Graph]
            List of Graph endpoints

        Returns
        -------
        trajs: List[Dict{'traj': List[tuple[Graph, GraphAction]]}]
           A list of trajectories.
        """
        return [{"traj": generate_forward_trajectory(i)} for i in graphs]

    def construct_batch(self, trajs, cond_info, log_rewards):
        """Construct a batch from a list of trajectories and their information

        Parameters
        ----------
        trajs: List[List[tuple[Graph, GraphAction]]]
            A list of N trajectories.
        cond_info: Tensor
            The conditional info that is considered for each trajectory. Shape (N, n_info)
        log_rewards: Tensor
            The transformed log-reward (e.g. torch.log(R(x) ** beta) ) for each trajectory. Shape (N,)
        Returns
        -------
        batch: gd.Batch
             A (CPU) Batch object with relevant attributes added
        """
        torch_graphs = [self.ctx.graph_to_Data(i[0], traj_len=k) for tj in trajs for k, i in enumerate(tj["traj"])]
        nx_graphs = [i[0] for tj in trajs for i in tj["traj"]]
        actions = [
            self.ctx.GraphAction_to_ActionIndex(g, a, fwd=True)
            for g, a in zip(torch_graphs, [i[1] for tj in trajs for i in tj["traj"]])
        ]
        batch = self.ctx.collate(torch_graphs)
        batch.traj_lens = torch.tensor([len(i["traj"]) for i in trajs])
        batch.log_p_B = torch.cat([i["bck_logprobs"] for i in trajs], 0)
        # batch.actions = torch.tensor(actions)
        batch.actions = actions
        batch.nx_graphs = nx_graphs
        batch.log_rewards = log_rewards
        batch.cond_info = cond_info
        batch.is_valid = torch.tensor([i.get("is_valid", True) for i in trajs]).float()

        # compute_batch_losses expects these two optional values, if someone else doesn't fill them in, default to 0
        batch.num_offline = 0
        batch.num_online = 0
        return batch

    def _update_lagged_model(self, model):

        # make a copy of the main model if the target network has not been initialized yet
        if self.lagged_model is None:
            import copy

            self.lagged_model = copy.deepcopy(model)

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1−τ) θ′
        target_net_state_dict = self.lagged_model.state_dict()
        policy_net_state_dict = model.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = (
                self.lagged_model_tau * policy_net_state_dict[key]
                + (1 - self.lagged_model_tau) * target_net_state_dict[key]
            )
        self.lagged_model.load_state_dict(target_net_state_dict)

    def compute_batch_losses(self, model: nn.Module, batch: gd.Batch, num_bootstrap: int = 0):
        """Compute the losses over trajectories contained in the batch

        Parameters
        ----------
        model: TrajectoryBalanceModel
           A GNN taking in a batch of graphs as input as per constructed by `self.construct_batch`.
           Must have a `logZ` attribute, itself a model, which predicts log of Z(cond_info)
        batch: gd.Batch
          batch of graphs inputs as per constructed by `self.construct_batch`
        num_bootstrap: int
          the number of trajectories for which the reward loss is computed. Ignored if 0."""
        dev = batch.x.device
        # A single trajectory is comprised of many graphs
        num_trajs = int(batch.traj_lens.shape[0])
        rewards = torch.exp(batch.log_rewards)
        cond_info = batch.cond_info

        if self.lagged_model_update_freq is not None and self.updates % self.lagged_model_update_freq == 0:
            self._update_lagged_model(model)

        # This index says which trajectory each graph belongs to, so
        # it will look like [0,0,0,0,1,1,1,2,...] if trajectory 0 is
        # of length 4, trajectory 1 of length 3, and so on.
        batch_idx = torch.arange(num_trajs, device=dev).repeat_interleave(batch.traj_lens)
        # The position of the last graph of each trajectory
        final_graph_idx = torch.cumsum(batch.traj_lens, 0) - 1

        # Forward pass of the model, returns a GraphActionCategorical and per object predictions
        # Here we will interpret the logits of the fwd_cat as Q values
        Q_tilde, per_state_preds = model(batch, cond_info[batch_idx])
        Q_tilde_lagged, _ = Q_tilde, (
            None if self.lagged_model is None else self.lagged_model(batch, cond_info[batch_idx])
        )

        """
        We learn Q_tilde(s,a) = Q(s,a)/alpha & V_tilde(s) = V(s)/alpha for which a simpler loss can be derived that only scales the reward (https://hackmd.io/@U-MDltE4Q_CnZYPu8is08A/rkRWbL8s0).
        The target value is Q_tilde(s,a) = r(s,a)/alpha + gamma * E_{s'|s,a}[V_tilde(s')]
        and the policy: pi(a|s) \propto exp( Q(s,a)/alpha ) = exp( Q_tilde(s,a) )
        """
        assert not self.do_q_prime_correction

        with torch.no_grad():
            if self.alpha < 0.0:
                raise ValueError("alpha must be positive")
            elif self.alpha > 0.0:
                rewards = rewards / self.alpha
                # V(s) = alpha * log sum_a exp( Q(s,a)/alpha ) --> V_tilde(s) = log sum_a exp( Q_tilde(s,a) )
                next_step_value = Q_tilde_lagged.logsumexp(
                    actions=batch.actions, nx_graphs=batch.nx_graphs, model=model
                ).detach()
            elif self.alpha == 0.0:
                # for standard Q-learning: V(s) = max_a Q(s,a)
                # next_step_value = Q_tilde.log_prob(batch.actions, nx_graphs=batch.nx_graphs, model=model, softmax=False)
                max_primary, max_secondary = Q_tilde._compute_batchwise_max()
                max_secondary = torch.nan_to_num(
                    max_secondary, nan=-float("inf")
                )  # TODO: not sure how to deal with masking here
                next_step_value = torch.maximum(max_primary, max_secondary)

            # We now need to compute the target, \hat Q = R_t + V_soft(s_t+1)
            # Shift t+1-> t, pad last state with a 0
            shifted_next_step_value = torch.cat([next_step_value[1:], torch.zeros_like(next_step_value[:1])])
            # Replace V(s_T). Since we've shifted the values in the array, V(s_T) is V(s_0)
            # of the next trajectory in the array, and rewards are terminal (0 except at s_T).
            shifted_next_step_value[final_graph_idx] = 0.0

            step_rewards = torch.zeros_like(shifted_next_step_value)
            step_rewards[final_graph_idx] = (
                rewards + (1 - batch.is_valid) * self.invalid_penalty
            )  # we only have a terminal reward

            hat_Q = step_rewards + self.gamma * shifted_next_step_value

        # Here were are again hijacking the GraphActionCategorical machinery to get Q[s,a], but
        # instead of logprobs we're just going to use the logits, i.e. the Q values.
        Q_sa = Q_tilde.log_prob(batch.actions, nx_graphs=batch.nx_graphs, model=model, softmax=False)

        losses = (Q_sa - hat_Q).pow(2)
        # losses = F.smooth_l1_loss(Q_sa, hat_Q, reduction='none')
        traj_losses = scatter(losses, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")
        loss = losses.mean()
        invalid_mask = 1 - batch.is_valid
        info = {
            "loss": loss.item(),
            "offline_loss": traj_losses[: batch.num_offline].mean() if batch.num_offline > 0 else 0,
            "online_loss": traj_losses[batch.num_offline :].mean() if batch.num_online > 0 else 0,
            "invalid_trajectories": invalid_mask.sum() / batch.num_online if batch.num_online > 0 else 0,
            "invalid_losses": (invalid_mask * traj_losses).sum() / (invalid_mask.sum() + 1e-4),
            "traj_lens": batch.traj_lens.float().mean(),
        }

        if not torch.isfinite(traj_losses).all():
            raise ValueError("loss is not finite")
        return loss, info

    def get_random_action_prob(self, it: int):
        return self.random_action_prob["eval"] if self.is_eval else self.random_action_prob["train"]

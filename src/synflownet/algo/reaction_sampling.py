import warnings
from typing import List, Optional

import torch
import torch.nn as nn
from rdkit import Chem
from torch import Tensor

from synflownet.algo.graph_sampling import Sampler
from synflownet.envs.graph_building_env import ActionIndex, Graph, GraphAction, GraphActionType, action_type_to_mask
from synflownet.envs.synthesis_building_env import ActionCategorical
from synflownet.models.graph_transformer import GraphTransformerSynGFN
from synflownet.utils.misc import get_worker_device

# ignore warnings
warnings.filterwarnings("ignore")


class SynthesisSampler(Sampler):
    """A helper class to sample from ActionCategorical-producing models"""

    def __init__(self, cfg, ctx, env, max_len, correct_idempotent=False, pad_with_terminal_state=False):
        """
        Parameters
        ----------
        env: ReactionTemplateEnv
            A reaction template environment.
        ctx: ReactionTemplateEnvContext
            A context.
        max_len: int
            If not None, ends trajectories of more than max_len steps.
        pad_with_terminal_state: bool
        """
        self.cfg = cfg
        self.ctx = ctx
        self.env = env
        self.max_len = max_len if max_len is not None else 5
        self.correct_idempotent = correct_idempotent
        self.pad_with_terminal_state = pad_with_terminal_state

    def sample_from_model(
        self,
        model: nn.Module,
        n: int,
        cond_info: Tensor,
        random_action_prob: float = 0.0,
        strict_forward_policy=None,
        use_argmax: bool = False,
    ):
        """Samples a model in a minibatch

        Parameters
        ----------
        model: nn.Module
            Model whose forward() method returns ActionCategorical instances
        n: int
            Number of graphs to sample
        cond_info: Tensor
            Conditional information of each trajectory, shape (n, n_info)
        random_action_prob: float
            Probability of taking a random action at each step

        Returns
        -------
        data: List[Dict]
           A list of trajectories. Each trajectory is a dict with keys
           - trajs: List[Tuple[Graph, Action]], the list of states and actions
           - fwd_logprob: sum logprobs P_F
           - bck_logprob: sum logprobs P_B
           - is_valid: is the generated graph valid according to the env & ctx
        """
        if strict_forward_policy is None:
            strict_forward_policy = self.cfg.algo.strict_forward_policy

        dev = get_worker_device()
        # This will be returned
        data = [
            {"traj": [], "reward_pred": None, "is_valid": True, "is_valid_bck": True, "is_sink": [], "bbs": []}
            for _ in range(n)
        ]
        bck_logprob: List[List[Tensor]] = [[] for _ in range(n)]

        graphs = [self.env.empty_graph() for _ in range(n)]
        done = [False] * n
        bck_a = [[GraphAction(GraphActionType.Stop)] for _ in range(n)]

        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]]

        for t in range(self.max_len):
            torch_graphs = [self.ctx.graph_to_Data(i, traj_len=t) for i in not_done(graphs)]
            nx_graphs = [g for g in not_done(graphs)]
            not_done_mask = torch.tensor(done, device=dev).logical_not()
            fwd_cat, *_, _ = model(self.ctx.collate(torch_graphs).to(dev), cond_info[not_done_mask])
            actions = fwd_cat.sample(
                nx_graphs=nx_graphs, model=model, random_action_prob=random_action_prob, use_argmax=use_argmax
            )
            graph_actions = [self.ctx.ActionIndex_to_GraphAction(g, a, fwd=True) for g, a in zip(torch_graphs, actions)]
            # Step each trajectory, and accumulate statistics
            for i, j in zip(not_done(range(n)), range(n)):
                data[i]["traj"].append((graphs[i], graph_actions[j]))
                if graph_actions[j].action in [
                    GraphActionType.AddFirstReactant,
                    GraphActionType.ReactBi,
                ]:  # for the bbs costs
                    data[i]["bbs"].append(graph_actions[j].bb)
                if graph_actions[j].action is GraphActionType.Stop:
                    done[i] = True
                    if not self.cfg.algo.tb.do_parameterize_p_b:
                        bck_logprob[i].append(torch.tensor([1.0], device=dev).log())
                    data[i]["is_sink"].append(1)
                    bck_a[i].append(GraphAction(GraphActionType.Stop))
                else:  # If not done, step the self.environment
                    gp = graphs[i]
                    gp = self.env.step(graphs[i], graph_actions[j])
                    try:
                        b_a = self.env.reverse(gp, graph_actions[j])
                        if strict_forward_policy:
                            bck_mol, both_are_bb, bb_idx = self.env.backward_step(gp, b_a)
                            if Chem.MolToSmiles(bck_mol) != Chem.MolToSmiles(self.ctx.graph_to_obj(graphs[i])):
                                if both_are_bb:
                                    if self.ctx.building_blocks[bb_idx] != Chem.MolToSmiles(
                                        self.ctx.graph_to_obj(graphs[i])
                                    ):
                                        raise ValueError(f"Reversing action does not yield the original molecule.")
                                else:
                                    raise ValueError(f"Reversing action does not yield the original molecule.")
                        bck_a[i].append(b_a)
                    except Exception as e:
                        warnings.warn(
                            f"Warning reversing action {graph_actions[j]}: {e}. \n Original mol: {Chem.MolToSmiles(self.ctx.graph_to_obj(graphs[i]))}"
                        )
                        bck_a[i].append(
                            GraphAction(GraphActionType.BckReactBi, rxn=graph_actions[j].rxn, bb=0)
                        )  # We need a bck_action to compute P_B
                        data[i]["is_valid"] = False
                        done[i] = True
                        if not self.cfg.algo.tb.do_parameterize_p_b:
                            bck_logprob[i].append(torch.tensor([1.0], device=dev).log())
                        data[i]["is_sink"].append(1)
                        continue
                    try:
                        Chem.SanitizeMol(gp)
                    except Exception as e:
                        warnings.warn(
                            f"Warning sanitizing molecule {Chem.MolToSmiles(gp)}: {e}. \n Action: {graph_actions[j]}, Original mol: {Chem.MolToSmiles(self.ctx.graph_to_obj(graphs[i]))}"
                        )
                        data[i]["is_valid"] = False  # Penalize RDKit errors
                        done[i] = True
                        if not self.cfg.algo.tb.do_parameterize_p_b:
                            bck_logprob[i].append(torch.tensor([1.0], device=dev).log())
                        data[i]["is_sink"].append(1)
                        continue
                    g = self.ctx.obj_to_graph(gp)

                    if not self.cfg.algo.tb.do_parameterize_p_b:
                        n_back = self.env.count_backward_transitions(
                            g, check_idempotent=self.correct_idempotent
                        )
                        if n_back > 0:
                            bck_logprob[i].append(
                                torch.tensor([1 / n_back], device=dev).log()
                            )
                        else:
                            bck_logprob[i].append(torch.tensor([0.001], device=dev).log())

                    if t == self.max_len - 1:
                        done[i] = True
                        data[i]["is_sink"].append(1)
                        continue

                    graphs[i] = g
                    data[i]["is_sink"].append(0)
                if done[i] and len(data[i]["traj"]) < 2:
                    data[i]["is_valid"] = False
            if all(done):
                break
        # is_sink indicates to a GFN algorithm that P_B(s) must be 1
        for i in range(n):
            data[i]["bck_logprob"] = sum(bck_logprob[i])
            if len(bck_logprob[i]):
                data[i]["bck_logprobs"] = torch.stack(bck_logprob[i]).reshape(-1)
            data[i]["result"] = graphs[i]
            data[i]["bck_a"] = bck_a[i]
            if self.pad_with_terminal_state:
                data[i]["traj"].append((graphs[i], GraphAction(GraphActionType.Stop)))
                data[i]["is_sink"].append(1)
        return data

    def sample_backward_from_graphs(
        self,
        graphs: List[Graph],
        model: Optional[nn.Module],
        cond_info: Optional[Tensor],
        random_action_prob: float = 0.0,
    ):
        """Sample a model's P_B starting from a list of graphs.

        Parameters
        ----------
        graphs: List[Graph]
            List of Graph endpoints
        model: nn.Module
            Model whose forward() method returns ActionCategorical instances
        cond_info: Tensor
            Conditional information of each trajectory, shape (n, n_info)
        random_action_prob: float
            Probability of taking a random action (only used if model parameterizes P_B)

        """
        dev = get_worker_device()
        n = len(graphs)
        done = [False] * n
        data = [
            {
                "traj": [(graphs[i], GraphAction(GraphActionType.Stop))],
                "is_valid": True,
                "is_valid_bck": True,
                "ends_in_s_0": True,
                "is_sink": [1],
                "bck_a": [GraphAction(GraphActionType.Stop)],
                "result": graphs[i],
                "bck_logprobs": [torch.tensor([1.0], device=dev).log()],
                "bbs": [],
            }
            for i in range(n)
        ]

        def not_done(lst):
            return [e for i, e in enumerate(lst) if not done[i]]

        if random_action_prob > 0:
            warnings.warn("Random action not implemented for backward sampling")

        t = 0
        while sum(done) < n and t < self.max_len:
            torch_graphs = [self.ctx.graph_to_Data(graphs[i], traj_len=t) for i in not_done(range(n))]
            bck_react_uni_masks = [torch.sum(g.bck_react_uni_mask) for g in torch_graphs]
            bck_react_bi_masks = [torch.sum(g.bck_react_bi_mask) for g in torch_graphs]
            remove_masks = [torch.sum(g.bck_remove_first_reactant_mask) for g in torch_graphs]
            masks_sum = [
                bck_react_uni_masks[i] + bck_react_bi_masks[i] + remove_masks[i] for i in range(len(torch_graphs))
            ]
            not_done_mask = torch.tensor(done, device=dev).logical_not()
            if model is not None:
                ci = cond_info[not_done_mask] if cond_info is not None else None
                _, bck_cat, *_ = model(self.ctx.collate(torch_graphs).to(dev), ci)
            else:
                gbatch = self.ctx.collate(torch_graphs)
                action_types = self.ctx.bck_action_type_order
                action_masks = [action_type_to_mask(t, gbatch, assert_mask_exists=True) for t in action_types]
                bck_cat = ActionCategorical(
                    gbatch,
                    raw_logits=[torch.ones_like(m) for m in action_masks],
                    keys=[GraphTransformerSynGFN.action_type_to_key(t) for t in action_types],
                    action_masks=action_masks,
                    types=action_types,
                )
            bck_actions = bck_cat.sample()
            graph_bck_actions = [
                self.ctx.ActionIndex_to_GraphAction(g, a, fwd=False) for g, a in zip(torch_graphs, bck_actions)
            ]

            for i, j in zip(not_done(range(n)), range(n)):
                if not done[i]:
                    g = graphs[i]
                    f_a = GraphAction(
                        GraphActionType.ReactUni, rxn=0
                    )  # "fake" fwd action, just to help with P_F computation for the tb loss
                    if masks_sum[j] == 0:
                        # if all masks are 0, we have reached the end of the trajectory
                        data[i]["is_valid_bck"] = False
                        done[i] = True
                        continue
                    b_a = graph_bck_actions[j]
                    data[i]["bck_logprobs"].append(torch.tensor([1.0], device=dev).log())
                    try:
                        gp, both_are_bb, bb_idx = self.env.backward_step(g, b_a)
                        if bb_idx is not None:
                            data[i]["bbs"].append(bb_idx)
                        if b_a.action == GraphActionType.BckRemoveFirstReactant:
                            data[i]["bbs"].append(
                                self.ctx.building_blocks.index(Chem.MolToSmiles(self.ctx.graph_to_obj(g)))
                            )
                        b_a.bb = both_are_bb  # set the bb flag in the action
                        if gp is None:
                            graphs[i] = self.env.empty_graph()
                        else:
                            graphs[i] = self.ctx.obj_to_graph(gp)
                        data[i]["traj"].append((graphs[i], f_a))
                        data[i]["bck_a"].append(b_a)
                        data[i]["is_sink"].append(0)
                    except Exception as e:
                        warnings.warn(
                            f"Warning running backward action {b_a}: {e}. \n Original mol: {Chem.MolToSmiles(self.ctx.graph_to_obj(g))}"
                        )
                        data[i]["is_valid_bck"] = False
                        done[i] = True
                        graphs[i] = self.env.empty_graph()
                        data[i]["traj"].append((graphs[i], f_a))
                        data[i]["bck_a"].append(b_a)
                        data[i]["is_sink"].append(0)
                        data[i]["bck_logprobs"].append(torch.tensor([1.0], device=dev).log())
                        continue
                    if len(graphs[i]) == 0:
                        done[i] = True
            t += 1

        for i in range(n):
            if (
                data[i]["bck_a"][-1].action != GraphActionType.BckRemoveFirstReactant
                or data[i]["is_valid_bck"] == False
            ):
                data[i]["ends_in_s_0"] = False
            data[i]["traj"] = data[i]["traj"][::-1]
            data[i]["bck_a"] = [GraphAction(GraphActionType.Stop)] + data[i]["bck_a"][::-1]
            data[i]["is_sink"] = data[i]["is_sink"][::-1]
            data[i]["bck_logprobs"] = torch.tensor(data[i]["bck_logprobs"][::-1], device=dev).reshape(-1)
            if self.pad_with_terminal_state:
                data[i]["traj"].append((graphs[i], GraphAction(GraphActionType.Stop)))
                data[i]["is_sink"].append(1)

        return data

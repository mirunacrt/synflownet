from itertools import chain

import torch
import torch.nn as nn
import torch_geometric.data as gd
import torch_geometric.nn as gnn
from torch_geometric.utils import add_self_loops

from gflownet.config import Config
from gflownet.envs.synthesis_building_env import ActionCategorical, ActionType


def mlp(n_in, n_hid, n_out, n_layer, act=nn.LeakyReLU):
    """Creates a fully-connected network with no activation after the last layer.
    If `n_layer` is 0 then this corresponds to `nn.Linear(n_in, n_out)`.
    """
    n = [n_in] + [n_hid] * n_layer + [n_out]
    return nn.Sequential(*sum([[nn.Linear(n[i], n[i + 1]), act()] for i in range(n_layer + 1)], [])[:-1])


torch.autograd.set_detect_anomaly(True)


class GraphTransformer(nn.Module):
    """An agnostic GraphTransformer class, and the main model used by other model classes

    This graph model takes in node features, edge features, and graph features (referred to as
    conditional information, since they condition the output). The graph features are projected to
    virtual nodes (one per graph), which are fully connected.

    The per node outputs are the final (post graph-convolution) node embeddings.

    The per graph outputs are the concatenation of a global mean pooling operation, of the final
    node embeddings, and of the final virtual node embeddings.
    """

    def __init__(self, x_dim, e_dim, g_dim, num_emb=64, num_layers=3, num_heads=2, num_noise=0, ln_type="pre"):
        """
        Parameters
        ----------
        x_dim: int
            The number of node features
        e_dim: int
            The number of edge features
        g_dim: int
            The number of graph-level features
        num_emb: int
            The number of hidden dimensions, i.e. embedding size. Default 64.
        num_layers: int
            The number of Transformer layers.
        num_heads: int
            The number of Transformer heads per layer.
        ln_type: str
            The location of Layer Norm in the transformer, either 'pre' or 'post', default 'pre'.
            (apparently, before is better than after, see https://arxiv.org/pdf/2002.04745.pdf)
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_noise = num_noise
        assert ln_type in ["pre", "post"]
        self.ln_type = ln_type

        self.x2h = mlp(x_dim + num_noise, num_emb, num_emb, 2)
        self.e2h = mlp(e_dim, num_emb, num_emb, 2)
        self.c2h = mlp(g_dim, num_emb, num_emb, 2)
        self.graph2emb = nn.ModuleList(
            sum(
                [
                    [
                        gnn.GENConv(num_emb, num_emb, num_layers=1, aggr="add", norm=None),
                        gnn.TransformerConv(num_emb * 2, num_emb, edge_dim=num_emb, heads=num_heads),
                        nn.Linear(num_heads * num_emb, num_emb),
                        gnn.LayerNorm(num_emb, affine=False),
                        mlp(num_emb, num_emb * 4, num_emb, 1),
                        gnn.LayerNorm(num_emb, affine=False),
                        nn.Linear(num_emb, num_emb * 2),
                    ]
                    for i in range(self.num_layers)
                ],
                [],
            )
        )

    def forward(self, g: gd.Batch, cond: torch.Tensor):
        """Forward pass

        Parameters
        ----------
        g: gd.Batch
            A standard torch_geometric Batch object. Expects `edge_attr` to be set.
        cond: torch.Tensor
            The per-graph conditioning information. Shape: (g.num_graphs, self.g_dim).

        Returns
        node_embeddings: torch.Tensor
            Per node embeddings. Shape: (g.num_nodes, self.num_emb).
        graph_embeddings: torch.Tensor
            Per graph embeddings. Shape: (g.num_graphs, self.num_emb * 2).
        """
        if self.num_noise > 0:
            x = torch.cat([g.x, torch.rand(g.x.shape[0], self.num_noise, device=g.x.device)], 1)
        else:
            x = g.x
        o = self.x2h(x)
        e = self.e2h(g.edge_attr)
        c = self.c2h(cond)
        num_total_nodes = g.x.shape[0]
        # Augment the edges with a new edge to the conditioning
        # information node. This new node is connected to every node
        # within its graph.
        # This is to integrate the conditioning information
        u, v = (
            torch.arange(num_total_nodes, device=o.device),
            g.batch + num_total_nodes,
        )  # u and v are used to create new edges in the augmented graph.
        # These edges connect each original node (u) to a new corresponding conditioning node (v).
        aug_edge_index = torch.cat([g.edge_index, torch.stack([u, v]), torch.stack([v, u])], 1)
        e_p = torch.zeros((num_total_nodes * 2, e.shape[1]), device=g.x.device)
        e_p[:, 0] = 1  # Manually create a bias term
        aug_e = torch.cat([e, e_p], 0)
        aug_edge_index, aug_e = add_self_loops(aug_edge_index, aug_e, "mean")
        aug_batch = torch.cat([g.batch, torch.arange(c.shape[0], device=o.device)], 0)

        # Append the conditioning information node embedding to o
        o = torch.cat([o, c], 0)
        for i in range(self.num_layers):
            # Run the graph transformer forward
            gen, trans, linear, norm1, ff, norm2, cscale = self.graph2emb[i * 7 : (i + 1) * 7]
            cs = cscale(c[aug_batch])
            if self.ln_type == "post":
                agg = gen(o, aug_edge_index, aug_e)
                l_h = linear(trans(torch.cat([o, agg], 1), aug_edge_index, aug_e))
                scale, shift = cs[:, : l_h.shape[1]], cs[:, l_h.shape[1] :]
                o = norm1(o + l_h * scale + shift, aug_batch)
                o = norm2(o + ff(o), aug_batch)
            else:
                o_norm = norm1(o, aug_batch)
                agg = gen(o_norm, aug_edge_index, aug_e)
                l_h = linear(trans(torch.cat([o_norm, agg], 1), aug_edge_index, aug_e))
                scale, shift = cs[:, : l_h.shape[1]], cs[:, l_h.shape[1] :]
                o = o + l_h * scale + shift
                o = o + ff(norm2(o, aug_batch))

        o_final = o[: -c.shape[0]]
        glob = torch.cat([gnn.global_mean_pool(o_final, g.batch), o[-c.shape[0] :]], 1)
        return o_final, glob


class GraphTransformerReactionsGFN(nn.Module):
    """GraphTransfomer class for a GFlowNet which outputs an ActionCategorical.

    Outputs logits corresponding to each action (template).
    """

    # The GraphTransformer outputs per-graph embeddings

    def __init__(
        self,
        env_ctx,
        cfg: Config,
        num_graph_out=1,
        do_bck=False,
    ) -> None:
        super().__init__()
        self.transf = GraphTransformer(
            x_dim=env_ctx.num_node_dim,
            e_dim=env_ctx.num_edge_dim,
            g_dim=env_ctx.num_cond_dim,
            num_emb=cfg.model.num_emb,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.graph_transformer.num_heads,
            ln_type=cfg.model.graph_transformer.ln_type,
        )
        self.env_ctx = env_ctx
        num_emb = cfg.model.num_emb
        num_glob_final = num_emb * 2  # *2 for concatenating global mean pooling & node embeddings
        self.action_type_order = env_ctx.action_type_order
        self.bck_action_type_order = env_ctx.bck_action_type_order

        # 3 Action types: ADD_REACTANT, REACT_UNI, REACT_BI
        # Every action type gets its own MLP that is fed the output of the GraphTransformer.
        # Here we define the number of inputs and outputs of each of those (potential) MLPs.
        self._action_type_to_num_inputs_outputs = {
            ActionType.Stop: (num_glob_final, 1),
            ActionType.AddFirstReactant: (num_glob_final, env_ctx.num_building_blocks),
            ActionType.AddReactant: (num_glob_final + env_ctx.num_bimolecular_rxns, env_ctx.num_building_blocks),
            ActionType.ReactUni: (num_glob_final, env_ctx.num_unimolecular_rxns),
            ActionType.ReactBi: (num_glob_final, env_ctx.num_bimolecular_rxns),
            ActionType.BckReactUni: (num_glob_final, env_ctx.num_unimolecular_rxns),
            ActionType.BckReactBi: (num_glob_final, env_ctx.num_bimolecular_rxns),
            ActionType.BckRemoveFirstReactant: (num_glob_final, 1),
        }

        self.add_reactant_hook = None

        self.do_bck = do_bck
        mlps = {}
        for atype in chain(self.action_type_order, self.bck_action_type_order if self.do_bck else []):
            num_in, num_out = self._action_type_to_num_inputs_outputs[atype]
            mlps[atype.cname] = mlp(num_in, num_emb, num_out, cfg.model.graph_transformer.num_mlp_layers)
        self.mlps = nn.ModuleDict(mlps)

        self.emb2graph_out = mlp(num_glob_final, num_emb, num_graph_out, cfg.model.graph_transformer.num_mlp_layers)
        self.logZ = mlp(env_ctx.num_cond_dim, num_emb * 2, 1, 2)

    def register_add_reactant_hook(self, hook):
        """
        Registers a custom hook for the AddReactant action.
        hook : callable
            The hook function to call with arguments (self, rxn_id, emb, g).
        """
        self.add_reactant_hook = hook

    def call_add_reactant_hook(self, rxn_id, emb, g):
        """
        Calls the registered hook for the AddReactant action, if any.
        rxn_id : int
            The ID of the reaction selected by the sampler.
        emb : torch.Tensor
            The embedding tensor for the current state.
        g : Graph
            The current graph.
        """
        if self.add_reactant_hook is not None:
            return self.add_reactant_hook(self, rxn_id, emb, g)
        else:
            raise RuntimeError("AddReactant hook not registered.")

    # ActionType.AddReactant gets masked in ActionCategorical class, not here
    def _action_type_to_mask(self, t, g):
        # if it is the first action, all logits get masked except those for AddFirstReactant
        # print(t.cname, getattr(g, "traj_len")[0])
        if hasattr(g, t.mask_name):
            masks = getattr(g, t.mask_name)
        att = []
        device = g.x.device
        for i in range(g.num_graphs):
            if getattr(g, "traj_len")[i] == 0 and t != ActionType.AddFirstReactant:
                att.append(torch.zeros(self._action_type_to_num_inputs_outputs[t][1]).to(device))
            elif getattr(g, "traj_len")[i] > 0 and t == ActionType.AddFirstReactant:
                att.append(torch.zeros(self._action_type_to_num_inputs_outputs[t][1]).to(device))
            else:
                att.append(
                    masks[
                        i
                        * self._action_type_to_num_inputs_outputs[t][1] : (i + 1)
                        * self._action_type_to_num_inputs_outputs[t][1]
                    ]
                    if hasattr(g, t.mask_name)
                    else torch.ones((self._action_type_to_num_inputs_outputs[t][1]), device=device)
                )
        att = torch.stack(att)
        return att.view(g.num_graphs, self._action_type_to_num_inputs_outputs[t][1]).to(device)

    def _action_type_to_logit(self, t, emb, g):
        logits = self.mlps[t.cname](emb)
        return self._mask(logits, self._action_type_to_mask(t, g))

    def _mask(self, x, m):
        # mask logit vector x with binary mask m, -1000 is a tiny log-value
        # Note to self: we can't use torch.inf here, because inf * 0 is nan (but also see issue #99)
        return x * m + -1000 * (1 - m)

    def _make_cat(self, g, emb, action_types, fwd):
        return ActionCategorical(
            g,
            emb,
            logits=[self._action_type_to_logit(t, emb, g) for t in action_types],
            masks=[self._action_type_to_mask(t, g) for t in action_types],
            types=action_types,
            fwd=fwd,
        )

    def forward(self, g: gd.Batch, cond: torch.Tensor, is_first_action: bool = False):
        """
        Forward pass of the GraphTransformerReactionsGFN.

        Parameters
        ----------
        g : gd.Batch
            A standard torch_geometric Batch object. Expects `edge_attr` to be set.
        cond : torch.Tensor
            The per-graph conditioning information. Shape: (g.num_graphs, self.g_dim).

        Returns
        -------
        ActionCategorical
        """
        _, graph_embeddings = self.transf(g, cond)
        graph_out = self.emb2graph_out(graph_embeddings)
        action_type_order = [a for a in self.action_type_order if a not in [ActionType.AddReactant]]
        # Map graph embeddings to action logits
        fwd_cat = self._make_cat(g, graph_embeddings, action_type_order, fwd=True)
        if self.do_bck:
            bck_cat = self._make_cat(g, graph_embeddings, self.bck_action_type_order, fwd=False)
            return fwd_cat, bck_cat, graph_out
        return fwd_cat, graph_out

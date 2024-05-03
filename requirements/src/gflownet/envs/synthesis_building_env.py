import enum
import math
import os
import pickle
import random
import re
import warnings
from functools import cached_property
from typing import List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd
from rdkit import Chem, RDLogger
from rdkit.Chem import BondType, ChiralType

from gflownet.synthesis_utils import Reaction
from gflownet.tasks.config import SEHReactionTaskConfig

logger = RDLogger.logger()
RDLogger.DisableLog("rdApp.*")

# Load templates and building blocks:
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
with open(os.path.join(repo_root, "data/building_blocks", SEHReactionTaskConfig.building_blocks_filename), "r") as file:
    BUILDING_BLOCKS = file.read().splitlines()
with open(os.path.join(repo_root, "data/templates", SEHReactionTaskConfig.templates_filename), "r") as file:
    TEMPLATES = file.read().splitlines()
with open(
    os.path.join(repo_root, "data/building_blocks", SEHReactionTaskConfig.precomputed_bb_masks_filename), "rb"
) as f:
    PRECOMPUTED_BB_MASKS = pickle.load(f)

DEFAULT_CHIRAL_TYPES = [ChiralType.CHI_UNSPECIFIED, ChiralType.CHI_TETRAHEDRAL_CW, ChiralType.CHI_TETRAHEDRAL_CCW]


class Graph(nx.Graph):
    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'<{list(self.nodes)}, {list(self.edges)}, {list(self.nodes[i]["v"] for i in self.nodes)}>'

    def bridges(self):
        return list(nx.bridges(self))

    def relabel_nodes(self, rmap):
        return nx.relabel_nodes(self, rmap)


class ActionType(enum.Enum):
    # Forward actions
    Stop = enum.auto()
    ReactUni = enum.auto()
    ReactBi = enum.auto()
    AddFirstReactant = enum.auto()
    AddReactant = enum.auto()
    # Backward actions
    BckReactUni = enum.auto()
    BckReactBi = enum.auto()
    BckRemoveFirstReactant = enum.auto()

    @cached_property
    def cname(self):
        return re.sub(r"(?<!^)(?=[A-Z])", "_", self.name).lower()

    @cached_property
    def mask_name(self):
        return self.cname + "_mask"

    @cached_property
    def is_backward(self):
        return self.name.startswith("Remove")


class ReactionTemplateEnvContext:
    """This context specifies how to create molecules by applying reaction templates."""

    def __init__(
        self,
        atoms: List[str] = [
            "C",
            "N",
            "O",
            "F",
            "P",
            "S",
            "Cl",
            "Br",
            "I",
            "B",
            "Sn",
            "Ca",
            "Na",
            "Ba",
            "Zn",
            "Rh",
            "Ag",
            "Li",
            "Yb",
            "K",
            "Fe",
            "Cs",
            "Bi",
            "Pd",
            "Cu",
            "Si",
        ],
        chiral_types: List = DEFAULT_CHIRAL_TYPES,
        charges: List[int] = [-3, -2, -1, 0, 1, 2, 3],
        expl_H_range: List[int] = [0, 1, 2, 3, 4],  # for N
        allow_explicitly_aromatic: bool = False,
        allow_5_valence_nitrogen: bool = False,
        num_cond_dim: int = 0,
        reaction_templates: List[str] = TEMPLATES,
        building_blocks: List[str] = BUILDING_BLOCKS,
        precomputed_bb_masks: np.ndarray = PRECOMPUTED_BB_MASKS,
    ):
        """An env context for generating molecules by sequentially applying reaction templates.
        Contains functionalities to build molecular graphs, create masks for actions, and convert molecules to other representations.

        Args:
            atoms (list): List of atom symbols.
            chiral_types (list): List of chiral types.
            charges (list): List of charges.
            expl_H_range (list): List of explicit H counts.
            allow_explicitly_aromatic (bool): Whether to allow explicitly aromatic molecules.
            allow_5_valence_nitrogen (bool): Whether to allow N with valence of 5.
            num_cond_dim (int): The dimensionality of the observations' conditional information vector (if >0)
            reaction_templates (list): List of SMIRKS.
            building_blocks (list): List of SMILES strings of building blocks.
            precomputed_bb_masks (np.ndarray): Precomputed masks (for bimoelcular reactions) for building blocks and reaction templates.
        """
        self.atom_attr_values = {
            "v": atoms + ["*"],
            "chi": chiral_types,
            "charge": charges,
            "expl_H": expl_H_range,
            "fill_wildcard": [None] + atoms,  # default is, there is nothing
        }
        self.atom_attrs = sorted(self.atom_attr_values.keys())
        self.atom_attr_slice = [0] + list(np.cumsum([len(self.atom_attr_values[i]) for i in self.atom_attrs]))
        self.allow_explicitly_aromatic = allow_explicitly_aromatic
        aromatic_optional = [BondType.AROMATIC] if allow_explicitly_aromatic else []
        self.bond_attr_values = {
            "type": [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE] + aromatic_optional,
        }
        self.bond_attrs = sorted(self.bond_attr_values.keys())
        self.bond_attr_slice = [0] + list(np.cumsum([len(self.bond_attr_values[i]) for i in self.bond_attrs]))
        self.default_wildcard_replacement = "C"
        self.negative_attrs = ["fill_wildcard"]
        pt = Chem.GetPeriodicTable()
        self._max_atom_valence = {
            **{a: max(pt.GetValenceList(a)) for a in atoms},
            # We'll handle nitrogen valence later explicitly in graph_to_Data
            "N": 3 if not allow_5_valence_nitrogen else 5,
            "*": 0,  # wildcard atoms have 0 valence until filled in
        }

        self.num_node_dim = sum(len(v) for v in self.atom_attr_values.values())
        self.num_edge_dim = sum(len(v) for v in self.bond_attr_values.values())
        self.num_cond_dim = num_cond_dim

        self.reactions = [Reaction(template=t) for t in reaction_templates]  # Reaction objects
        self.unimolecular_reactions = [r for r in self.reactions if r.num_reactants == 1]  # rdKit reaction objects
        self.bimolecular_reactions = [r for r in self.reactions if r.num_reactants == 2]
        self.num_unimolecular_rxns = len(self.unimolecular_reactions)
        self.num_bimolecular_rxns = len(self.bimolecular_reactions)

        self.building_blocks = building_blocks
        self.building_blocks_mols = [Chem.MolFromSmiles(bb) for bb in building_blocks]
        self.num_building_blocks = len(building_blocks)
        self.precomputed_bb_masks = precomputed_bb_masks

        # Order in which models have to output logits
        self.action_type_order = [
            ActionType.Stop,
            ActionType.ReactUni,
            ActionType.ReactBi,
            ActionType.AddReactant,
            ActionType.AddFirstReactant,
        ]  # ActionType.AddReactant used separately in a hook during sampling
        self.bck_action_type_order = [
            ActionType.BckReactUni,
            ActionType.BckReactBi,
            ActionType.BckRemoveFirstReactant,
        ]  # (0, j, None), (1, j, 0) or (1, j, 1), (2, None, None)

    def aidx_to_action_type(self, aidx: Tuple[int, int, Optional[int]], fwd: bool = True):
        if fwd:
            action_type_order = self.action_type_order
        else:
            action_type_order = self.bck_action_type_order
        return action_type_order[aidx[0]]

    def action_type_to_aidx(self, action_type: ActionType, fwd: bool = True):
        if fwd:
            action_type_order = self.action_type_order
        else:
            action_type_order = self.bck_action_type_order
        return action_type_order.index(action_type)

    def create_masks(self, smi: Union[str, Chem.Mol, Graph], fwd: bool = True, unimolecular: bool = True) -> List[int]:
        """Creates masks for reaction templates for a given molecule.

        Args:
            mol (Chem.Mol): Molecule as a rdKit Mol object.
            fwd (bool): Whether it is a forward or a backward step.
            unimolecular (bool): Whether it is a unimolecular or a bimolecular reaction.

        Returns:
            (torch.Tensor): Masks for invalid actions.
        """
        mol = self.get_mol(smi)
        Chem.SanitizeMol(mol)
        if unimolecular:
            masks = np.ones(self.num_unimolecular_rxns)
            reactions = self.unimolecular_reactions
        else:
            masks = np.ones(self.num_bimolecular_rxns)
            reactions = self.bimolecular_reactions
        for idx, r in enumerate(reactions):
            if fwd:
                if not r.is_reactant(mol):
                    masks[idx] = 0
            else:
                mol_copy = Chem.Mol(mol)
                mol_copy = Chem.MolFromSmiles(Chem.MolToSmiles(mol_copy))
                Chem.Kekulize(mol_copy, clearAromaticFlags=True)
                if not (r.is_product(mol) or r.is_product(mol_copy)):
                    masks[idx] = 0
        return masks

    def create_masks_for_bb(self, smi: Union[str, Chem.Mol, Graph], bimolecular_rxn_idx: int) -> List[bool]:
        """Create masks for building blocks for a given molecule."""
        mol = self.get_mol(smi)
        Chem.SanitizeMol(mol)
        reaction = self.bimolecular_reactions[bimolecular_rxn_idx]
        reactants = reaction.rxn.GetReactants()
        assert mol.HasSubstructMatch(reactants[0]) or mol.HasSubstructMatch(
            reactants[1]
        ), "Molecule does not match reaction template -- this should be verified at the reaction-selection step."

        masks = np.zeros(self.num_building_blocks)
        for idx, bb in enumerate(self.building_blocks_mols):
            fit1 = mol.HasSubstructMatch(reactants[0]) and bb.HasSubstructMatch(reactants[1])
            fit2 = mol.HasSubstructMatch(reactants[1]) and bb.HasSubstructMatch(reactants[0])
            if fit1 or fit2:
                masks[idx] = 1.0
        return masks

    def create_masks_for_bb_from_precomputed(
        self, smi: Union[str, Chem.Mol, Graph], bimolecular_rxn_idx: int
    ) -> List[bool]:
        """Creates masks for building blocks (for the 2nd reactant) for a given molecule and bimolecular reaction.
        Uses masks precomputed with data/building_blocks/precompute_bb_masks.py.

        Args:
            smi (Union[str, Chem.Mol, Graph]): Molecule as a rdKit Mol object.
            bimolecular_rxn_idx (int): Index of the bimolecular reaction.
        """
        mol = self.get_mol(smi)
        Chem.SanitizeMol(mol)
        reaction = self.bimolecular_reactions[bimolecular_rxn_idx]
        reactants = reaction.rxn.GetReactants()

        precomputed_bb_masks = self.precomputed_bb_masks[:, bimolecular_rxn_idx]
        mol_mask = np.array(
            [  # we reverse the order of the reactants w.r.t BBs (i.e. reactants[1] first)
                np.ones((self.num_building_blocks,)) * float(mol.HasSubstructMatch(reactants[1])),
                np.ones((self.num_building_blocks,)) * float(mol.HasSubstructMatch(reactants[0])),
            ]
        )
        masks = np.max(mol_mask * precomputed_bb_masks, axis=0).astype(np.float64)
        return masks

    def get_mol(self, smi: Union[str, Chem.Mol, Graph]) -> Chem.Mol:
        """
        A function that returns an `RDKit.Chem.Mol` object.

        Args:
            smi (str or RDKit.Chem.Mol or Graph): The query molecule, as either a
                SMILES string an `RDKit.Chem.Mol` object, or a Graph.

        Returns:
            RDKit.Chem.Mol
        """
        if isinstance(smi, str):
            return Chem.MolFromSmiles(smi, replacements={"[2H]": "[H]"})
        elif isinstance(smi, Chem.Mol):
            return smi
        elif isinstance(smi, Graph):
            return self.graph_to_mol(smi)
        else:
            raise TypeError(f"{type(smi)} not supported, only `str` or `rdkit.Chem.Mol`")

    def mol_to_graph(self, mol: Chem.Mol) -> Graph:
        """Convert an RDMol to a Graph"""
        g = Graph()
        mol = Chem.Mol(mol)  # Make a copy
        mol.UpdatePropertyCache()
        if not self.allow_explicitly_aromatic:
            # If we disallow aromatic bonds, ask rdkit to Kekulize mol and remove aromatic bond flags
            Chem.Kekulize(mol, clearAromaticFlags=True)
        # Only set an attribute tag if it is not the default attribute
        for a in mol.GetAtoms():
            attrs = {
                "atomic_number": a.GetAtomicNum(),
                "chi": a.GetChiralTag(),
                "charge": a.GetFormalCharge(),
                "expl_H": a.GetNumExplicitHs(),
            }
            g.add_node(
                a.GetIdx(),
                v=a.GetSymbol(),
                **{attr: val for attr, val in attrs.items()},
                **({"fill_wildcard": None} if a.GetSymbol() == "*" else {}),
            )
        for b in mol.GetBonds():
            attrs = {"type": b.GetBondType()}
            g.add_edge(
                b.GetBeginAtomIdx(),
                b.GetEndAtomIdx(),
                **{attr: val for attr, val in attrs.items()},
            )
        return g

    def graph_to_mol(self, g: Graph) -> Chem.Mol:
        """Convert a Graph to an RDKit Mol"""
        mp = Chem.RWMol()
        mp.BeginBatchEdit()
        for i in range(len(g.nodes)):
            d = g.nodes[i]
            s = d.get("fill_wildcard", d["v"])
            a = Chem.Atom(s if s is not None else self.default_wildcard_replacement)
            if "chi" in d:
                a.SetChiralTag(d["chi"])
            if "charge" in d:
                a.SetFormalCharge(d["charge"])
            if "expl_H" in d:
                a.SetNumExplicitHs(d["expl_H"])
            if "no_impl" in d:
                a.SetNoImplicit(d["no_impl"])
            mp.AddAtom(a)
        for e in g.edges:
            d = g.edges[e]
            mp.AddBond(e[0], e[1], d.get("type", BondType.SINGLE))
        mp.CommitBatchEdit()
        Chem.SanitizeMol(mp)
        return Chem.MolFromSmiles(Chem.MolToSmiles(mp))

    def graph_to_Data(self, g: Graph, traj_len: int) -> gd.Data:
        """Convert a networkx Graph to a torch geometric Data instance"""
        x = np.zeros((max(1, len(g.nodes)), self.num_node_dim))
        x[0, -1] = len(g.nodes) == 0  # If there are no nodes, set the last dimension to 1

        for i, n in enumerate(g.nodes):
            ad = g.nodes[n]
            for k, sl in zip(self.atom_attrs, self.atom_attr_slice):
                idx = self.atom_attr_values[k].index(ad[k]) if k in ad else 0
                x[i, sl + idx] = 1  # One-hot encode the attribute value

        edge_attr = np.zeros((len(g.edges) * 2, self.num_edge_dim))
        for i, e in enumerate(g.edges):
            ad = g.edges[e]
            for k, sl in zip(self.bond_attrs, self.bond_attr_slice):
                idx = self.bond_attr_values[k].index(ad[k])
                edge_attr[i * 2, sl + idx] = 1
                edge_attr[i * 2 + 1, sl + idx] = 1
        edge_index = np.array([e for i, j in g.edges for e in [(i, j), (j, i)]]).reshape((-1, 2)).T.astype(np.int64)

        data = dict(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            # add attribute for masks
            react_uni_mask=self.create_masks(g, fwd=True, unimolecular=True),
            react_bi_mask=self.create_masks(g, fwd=True, unimolecular=False),
            bck_react_uni_mask=self.create_masks(g, fwd=False, unimolecular=True),
            bck_react_bi_mask=self.create_masks(g, fwd=False, unimolecular=False),
            traj_len=np.array(
                [traj_len]
            ),  # if traj_len is 0, the only possible action is AddFirstReactant; all other actions are masked
        )
        data = gd.Data(**{k: torch.from_numpy(v) for k, v in data.items()})
        data.edge_attr = data.edge_attr.to(torch.float32)
        data.traj_len = data.traj_len.to(torch.int32)
        data.x = data.x.to(torch.float32)
        return data

    def collate(self, graphs: List[gd.Data]):
        """Batch Data instances"""
        return gd.Batch.from_data_list(graphs, follow_batch=["edge_index"])

    def object_to_log_repr(self, g: Graph):
        """Convert a Graph to a string representation"""
        try:
            mol = self.graph_to_mol(g)
            assert mol is not None
            return Chem.MolToSmiles(mol)
        except Exception:
            return ""

    def traj_to_log_repr(self, traj: List[Tuple[Graph]]):
        """Convert a tuple of graph, action idx to a string representation, action idx"""
        smi_traj = []
        for i in traj:
            mol = self.graph_to_mol(i[0])
            assert mol is not None
            smi_traj.append((Chem.MolToSmiles(mol), i[1]))
        return str(smi_traj)


class ReactionTemplateEnv:
    """Molecules and reaction templates environment. The new, initial states are Enamine building block molecules.

    This environment specifies how to obtain new molecules from applying reaction templates to current molecules. Works by
    having the agent select a reaction template. Masks ensure that only valid templates are selected.
    """

    def __init__(self, reaction_templates: List[str] = TEMPLATES, building_blocks: List[str] = BUILDING_BLOCKS):
        """A reaction template environment instance"""
        self.ctx = ReactionTemplateEnvContext(reaction_templates=reaction_templates, building_blocks=building_blocks)

    def new(self) -> Graph:
        smi = random.choice(self.ctx.building_blocks)
        mol = self.ctx.get_mol(smi)
        return self.ctx.mol_to_graph(mol)

    def empty_graph(self) -> Graph:
        return Graph()

    def step(self, smi: Union[str, Chem.Mol, Graph], action: Tuple[int, int, Optional[int]]) -> Chem.Mol:
        """Applies the action to the current state and returns the next state.

        Args:
            mol (Chem.Mol): Current state as an RDKit mol.
            action Tuple[int, int, Optional[int]]: Action indices to apply to the current state.
            (ActionType, reaction_template_idx, reactant_idx)

        Returns:
            (Chem.Mol): Next state as an RDKit mol.
        """
        mol = self.ctx.get_mol(smi)
        Chem.SanitizeMol(mol)
        if self.ctx.aidx_to_action_type(action) == ActionType.Stop:
            return mol
        elif (
            self.ctx.aidx_to_action_type(action) == ActionType.AddReactant
            or self.ctx.aidx_to_action_type(action) == ActionType.AddFirstReactant
        ):
            return self.ctx.get_mol(self.ctx.building_blocks[action[1]])
        elif self.ctx.aidx_to_action_type(action) == ActionType.ReactUni:
            reaction = self.ctx.unimolecular_reactions[action[1]]
            p = reaction.run_reactants((mol,))
            return p
        else:
            reaction = self.ctx.bimolecular_reactions[action[1]]
            reactant2 = self.ctx.get_mol(self.ctx.building_blocks[action[2]])
            p = reaction.run_reactants((mol, reactant2))
            return p

    def backward_step(self, smi: Union[str, Chem.Mol, Graph], action: Tuple[int, int]) -> Chem.Mol:
        """Applies the action to the current state and returns the previous (parent) state.

        Args:
            mol (Chem.Mol): Current state as an RDKit Mol object.
            action: Tuple[int, int]: Backward action indices to apply to the current state.
            (ActionType, reaction_template_idx)

        Returns:
            (Chem.Mol): Previous state as an RDKit mol and if the reaction is bimolecular,
            returns whether both products (reactants when fwd) are building blocks.
            This is important because if they are, we need to randomly select which to keep
            and this p_B of this action = 1/2.
        """
        mol = self.ctx.get_mol(smi)
        # Chem.SanitizeMol(mol)
        if self.ctx.aidx_to_action_type(action, fwd=False) == ActionType.BckRemoveFirstReactant:
            return self.ctx.get_mol(""), None
        elif self.ctx.aidx_to_action_type(action, fwd=False) == ActionType.BckReactUni:
            reaction = self.ctx.unimolecular_reactions[action[1]]
            return reaction.run_reverse_reactants((mol,)), None  # return the product and None (no reactant was removed)
        else:  # if bimolecular
            reaction = self.ctx.bimolecular_reactions[action[1]]
            products = reaction.run_reverse_reactants((mol,))
            products_smi = [Chem.MolToSmiles(p) for p in products]

            both_are_bb = 0
            # If both products are building blocks, randomly select which to keep
            if (products_smi[0] in self.ctx.building_blocks) and (products_smi[1] in self.ctx.building_blocks):
                both_are_bb = 1
                selected_product = random.choice(products)
            elif products_smi[0] in self.ctx.building_blocks:
                selected_product = products[1]
            elif products_smi[1] in self.ctx.building_blocks:
                selected_product = products[0]
            elif len(products_smi[0]) > len(products_smi[1]):
                selected_product = products[0]
            else:
                selected_product = products[1]

            try:
                rw_mol = Chem.RWMol(selected_product)
            except:
                print(action[0], action[1], Chem.MolToSmiles(mol))
            atoms_to_remove = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetSymbol() == "*"]
            for idx in sorted(
                atoms_to_remove, reverse=True
            ):  # Remove atoms in reverse order to avoid reindexing issues
                rw_mol.ReplaceAtom(idx, Chem.Atom("H"))
            atoms_to_remove = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetSymbol() == "[CH]"]
            for idx in sorted(
                atoms_to_remove, reverse=True
            ):  # Remove atoms in reverse order to avoid reindexing issues
                rw_mol.ReplaceAtom(idx, Chem.Atom("C"))
            try:
                rw_mol.UpdatePropertyCache()
            except Chem.rdchem.AtomValenceException as e:
                warnings.warn(f"{e}: Reaction {reaction.template}, product {Chem.MolToSmiles(selected_product)}")
            return rw_mol, both_are_bb

    def parents(self, smi: Union[str, Chem.Mol, Graph]) -> List[Chem.Mol]:
        """Returns the parent molecules of a given molecule.

        Args:
            mol (Chem.Mol): Molecule as an RDKit mol.

        Returns:
            (list): List of parent molecules as RDKit mols.
        """
        mol = self.ctx.get_mol(smi)
        parents = []
        for i, reaction in enumerate(self.ctx.unimolecular_reactions):
            # mol.UpdatePropertyCache()
            Chem.SanitizeMol(mol)
            if reaction.is_product(mol):
                parents.append(self.backward_step(mol, (0, i)))
            Chem.SanitizeMol(mol)
        for i, reaction in enumerate(self.ctx.bimolecular_reactions):
            # mol.UpdatePropertyCache()
            Chem.SanitizeMol(mol)
            if reaction.is_product(mol):
                parents.append(self.backward_step(mol, (1, i)))
            Chem.SanitizeMol(mol)
        return parents

    def parents_count(self, smi: Union[str, Chem.Mol, Graph]) -> int:
        """Returns the number of parent molecules of a given molecule.

        Args:
            mol (Chem.Mol): Molecule as an RDKit mol.

        Returns:
            (int): Number of parents.
        """
        mol = self.ctx.get_mol(smi)
        parents_count = 0
        for i, rxn in enumerate(self.ctx.unimolecular_reactions):
            # mol.UpdatePropertyCache()
            Chem.SanitizeMol(mol)
            if rxn.is_product(mol):
                parents_count += 1
        for i, reaction in enumerate(self.ctx.bimolecular_reactions):
            # mol.UpdatePropertyCache()
            Chem.SanitizeMol(mol)
            if reaction.is_product(mol):
                parents_count += 1
        return parents_count

    def count_backward_transitions(self, smi: Union[str, Chem.Mol, Graph]) -> int:
        """Counts the number of backward transitions from a given molecule.

        Args:
            mol (Chem.Mol): Molecule as an RDKit mol.

        Returns:
            (int): Number of possible backward transitions.
        """
        mol = self.ctx.get_mol(smi)
        return self.parents_count(mol)


class ActionCategorical:
    def __init__(
        self,
        graphs: gd.Batch,
        graph_embeddings: torch.Tensor,
        logits: List[torch.Tensor],
        types: List[ActionType],
        masks: List[torch.Tensor] = None,
        fwd: bool = True,
    ):
        """A categorical distribution over the actions.

        Parameters:
        graphs: Batch
            A batch of graphs to which the logits correspond.
        graph_embeddings: Tensor
            A tensor of shape (n, m) where n is the number of graphs and m is the embedding dimension.
        logits: List[Tensor]
            A list of tensors, each of length (n,m) - n is number of graphs and
            there are m possible actions per action type
            The length of the `logits` list is equal to the number of action
            types available.
        types: List[ActionType]
            The action type each logit group corresponds to.
        masks: List[Tensor], default=None
            If not None, a list of broadcastable tensors that multiplicatively
            mask out logits of invalid actions
        fwd: bool, default=True
            Whether the action space is for forward or backward actions.
        """
        self.graphs = graphs
        self.graph_embeddings = graph_embeddings
        self.graphs_list = graphs.to_data_list()
        self.dev = dev = graphs.x.device
        self.ctx = ReactionTemplateEnvContext()
        self.num_graphs = graphs.num_graphs
        if masks is not None:
            assert len(logits) == len(masks)
        self._epsilon = 1e-38
        self.masks = masks if masks is not None else [torch.ones_like(l) for l in logits]
        self.logprobs = None
        self.batch = torch.arange(graphs.num_graphs, device=dev)
        self.fwd = fwd

        # For fwd actions, there is a hierarchy of action types: AddFirstReactant, Stop, UniReact, BiReact to be sampled first, then AddReactant
        # The logits are in the order: Stop, UniReact, BiReact, AddReactant
        self.action_hierarchy = {
            "fwd": {
                "primary": types,
                "secondary": [ActionType.AddReactant],
            },
            "bck": {
                "primary": types,
            },
        }
        logits.append(
            torch.zeros((self.num_graphs, self.ctx.num_building_blocks), device=dev)
        )  # Placeholder for AddReactant logits
        self.logits = logits
        if self.fwd:
            self.action_type_to_logits_index = {
                action_type: i for i, action_type in enumerate(types + [ActionType.AddReactant])
            }
        else:
            self.action_type_to_logits_index = {action_type: i for i, action_type in enumerate(types)}

        self.primary_logits = self.get_primary_logits()
        self.secondary_logits = self.get_secondary_logits()

    def get_logits_for_action_type(self, action_type):
        """Retrieve logits for a given action type."""
        index = self.action_type_to_logits_index.get(action_type)
        if index is not None:
            return self.logits[index]
        else:
            raise ValueError(f"Invalid action type: {action_type}")

    def get_primary_logits(self):
        """Retrieve logits for primary actions based on the current mode (fwd or bck)."""
        key = "fwd" if self.fwd else "bck"
        primary_action_types = self.action_hierarchy[key]["primary"]
        return [self.get_logits_for_action_type(action_type) for action_type in primary_action_types]

    def get_secondary_logits(self):
        """Retrieve logits for secondary actions, if any, based on the current mode (fwd or bck)."""
        key = "fwd" if self.fwd else "bck"
        if "secondary" in self.action_hierarchy[key]:
            secondary_action_types = self.action_hierarchy[key]["secondary"]
            return [self.get_logits_for_action_type(action_type) for action_type in secondary_action_types]
        return []

    def _compute_batchwise_max(
        self,
    ):
        """Compute the argmax for each batch element in the batch of logits.

        Parameters
        ----------

        Returns
        -------
        overall_max_per_graph: Tensor
            A tensor of shape (n,m) where n is the number of graphs in the batch.
            Each element is the max value of the logits for the corresponding graph.
            m is 1 if there is one hierarchy of actions, and 2 if there are two hierarchies.
        """
        primary_logits = self.primary_logits
        secondary_logits = self.secondary_logits

        # Compute max for primary logits
        max_per_primary_type = [torch.max(tensor, dim=1)[0] for tensor in primary_logits]
        overall_max_per_graph_primary, _ = torch.max(torch.stack(max_per_primary_type), dim=0)

        # Compute max for secondary logits if they exist
        if secondary_logits:
            max_per_secondary_type = [torch.max(tensor, dim=1)[0] for tensor in secondary_logits]
            overall_max_per_graph_secondary, _ = torch.max(torch.stack(max_per_secondary_type), dim=0)
            overall_max_per_graph = torch.stack((overall_max_per_graph_primary, overall_max_per_graph_secondary))
        else:
            overall_max_per_graph = overall_max_per_graph_primary

        return overall_max_per_graph

    def argmax(
        self,
        x: List[torch.Tensor],
    ):
        max_per_type = [
            torch.max(tensor, dim=1) for tensor in x
        ]  # for each graph in batch and for each action type, get max value and index
        max_values_per_type = [pair[0] for pair in max_per_type]
        argmax_indices_per_type = [pair[1] for pair in max_per_type]
        _, type_indices = torch.max(torch.stack(max_values_per_type), dim=0)
        action_indices = torch.gather(torch.stack(argmax_indices_per_type), 0, type_indices.unsqueeze(0)).squeeze(0)
        argmax_pairs = list(zip(type_indices.tolist(), action_indices.tolist()))  # action type, action idx
        return argmax_pairs

    def logsoftmax(self):
        """Compute log-probabilities given logits"""
        if self.logprobs is not None:
            return self.logprobs
        # we need to compute the log-probabilities (1) for the primary logits and (2) for the secondary logits
        primary_logits = self.primary_logits
        secondary_logits = self.secondary_logits
        max_logits = self._compute_batchwise_max()
        if secondary_logits:
            max_logits_primary, max_logits_secondary = max_logits
        else:
            max_logits_primary = max_logits
            max_logits_secondary = None

        # correct primary logits by max and exponentiate
        corr_logits_primary = [tensor - max_logits_primary.view(-1, 1) for tensor in primary_logits]
        exp_logits_primary = [i.exp().clamp(self._epsilon) for i in corr_logits_primary]
        # compute logZ for primary logits
        merged_exp_logits_primary = torch.cat(exp_logits_primary, dim=1)
        log_Z_primary = merged_exp_logits_primary.sum(dim=1).log()
        # compute log-probabilities for primary logits
        log_probs = [l - log_Z_primary.view(-1, 1) for l in corr_logits_primary]
        # if there are secondary logits, compute log-probabilities for them
        if max_logits_secondary is not None:
            corr_logits_secondary = [tensor - max_logits_secondary.view(-1, 1) for tensor in secondary_logits]
            exp_logits_secondary = [i.exp().clamp(self._epsilon) for i in corr_logits_secondary]
            merged_exp_logits_secondary = torch.cat(exp_logits_secondary, dim=1)
            log_Z_secondary = merged_exp_logits_secondary.sum(dim=1).log()
            log_probs.append(torch.cat(corr_logits_secondary, dim=1) - log_Z_secondary.view(-1, 1))
        return log_probs

    def add_reactant_hook(self, model, rxn_id, emb, g):
        """
        The hook function to be called for the AddReactant action.
        Parameters
        model : GraphTransformerReactionsGFN
            The model instance.
        rxn_id : int
            The ID of the reaction selected by the sampler.
        emb : torch.Tensor
            The embedding tensor for the current state.
        g : Graph
            The current graph.

        Returns
        torch.Tensor
            The logits or output of the MLP after being called with the expanded input.
        """
        # Convert `rxn_id` to a one-hot vector
        rxn_features = torch.zeros(model.env_ctx.num_bimolecular_rxns).to(emb.device)
        rxn_features[rxn_id] = 1
        expanded_input = torch.cat((emb, rxn_features), dim=-1)
        return model.mlps[ActionType.AddReactant.cname](expanded_input)

    def sample(self, traj_len: Optional[int], nx_graphs: List[nx.Graph] = None, model: nn.Module = None):
        """Samples from the categorical distribution"""
        primary_logits = self.primary_logits
        # The first action in a trajectory is always AddFirstReactant (select a building block)
        if traj_len == 0:
            noise = torch.rand(
                primary_logits[self.action_type_to_logits_index[ActionType.AddFirstReactant]].shape, device=self.dev
            )
            gumbel = primary_logits[0] - (-noise.log()).log()
            argmax = self.argmax(x=[gumbel])
            action_type = self.ctx.action_type_to_aidx(ActionType.AddFirstReactant)
            return [(action_type, a[1], None) for a in argmax]
        if traj_len == 1:
            # we ensure that a Stop action does not get sampled right at the beginning of the trajectory
            stop_action_idx = self.action_type_to_logits_index[ActionType.Stop]
            stop_action_logits = self.primary_logits[
                stop_action_idx
            ]  # just in case a stop action is required because of invalid bb and we need to replace the mask with original value
            primary_logits[stop_action_idx] = torch.zeros_like(self.primary_logits[stop_action_idx]) - torch.inf
            self.primary_logits[stop_action_idx] = torch.zeros_like(self.primary_logits[stop_action_idx]) - 1000.0
        # Use the Gumbel trick to sample categoricals
        u = [torch.rand(i.shape, device=self.dev) for i in primary_logits]
        gumbel = [logit - (-noise.log()).log() for logit, noise in zip(primary_logits, u)]
        argmax = self.argmax(x=gumbel)  # tuple of action type, action idx
        for i, t in enumerate(argmax):
            if self.ctx.aidx_to_action_type(t, fwd=self.fwd) == ActionType.Stop:
                argmax[i] = (0, None, None)  # argmax returns 0 for ActionIdx
            elif self.ctx.aidx_to_action_type(t, fwd=self.fwd) in [ActionType.ReactUni, ActionType.BckReactUni]:
                argmax[i] = t + (None,)  # pad with None
            elif self.ctx.aidx_to_action_type(t, fwd=self.fwd) == ActionType.ReactBi:  # sample reactant
                masks = torch.tensor(self.ctx.create_masks_for_bb_from_precomputed(nx_graphs[i], t[1]), device=self.dev)
                if torch.all(masks == 0.0):
                    if traj_len == 1:
                        self.primary_logits[0][i] = stop_action_logits[i]
                    argmax[i] = (0, None, None)
                    continue
                # Call the hook to get the logits for the AddReactant action
                model.register_add_reactant_hook(self.add_reactant_hook)
                add_reactant_logits = model.call_add_reactant_hook(t[1], self.graph_embeddings[i], self.graphs[i])
                masked_logits = torch.zeros_like(add_reactant_logits) - torch.inf
                masked_logits[masks.bool()] = add_reactant_logits[masks.bool()]
                device = masked_logits.device
                self.secondary_logits[0][i] = torch.where(
                    masked_logits == -torch.inf, torch.tensor(-1000.0).to(device), masked_logits
                )
                noise = torch.rand(masked_logits.shape, device=self.dev)
                gumbel = masked_logits - (-noise.log()).log()
                max_idx = int(gumbel.argmax())
                assert masks[max_idx] == 1.0, "This index should not be masked"
                argmax[i] = t + (max_idx,)
            # else: # TODO Action type BckReactBi
        return argmax

    def log_prob(
        self,
        actions: List[Tuple[int, int, int]],
        traj_idcs: Optional[torch.tensor] = None,
        nx_graphs: Optional[List[nx.Graph]] = None,
        model: Optional[nn.Module] = None,
    ):
        """Access the log-probability of actions"""
        # Initialize a tensor to hold the log probabilities for each action
        if self.fwd:
            for i, (action, traj_idx) in enumerate(zip(actions, traj_idcs)):
                action_type, action_idx, action2_idx = action
                # Instances where we've changed the logits values during sampling:
                if (
                    traj_idx == 1 and self.ctx.aidx_to_action_type(action, fwd=self.fwd) != ActionType.Stop
                ):  # if the trajectory is of length 1 we have masked stop actions;
                    # but we have unmasked them if there was no other action available (line 804)
                    stop_action_idx = self.action_type_to_logits_index[ActionType.Stop]
                    self.primary_logits[stop_action_idx][i] = (
                        torch.zeros_like(self.primary_logits[stop_action_idx][i]) - 1000.0
                    )
                if (
                    self.ctx.aidx_to_action_type(action, fwd=self.fwd) == ActionType.ReactBi
                ):  # secondary logits were computed
                    masks = torch.tensor(
                        self.ctx.create_masks_for_bb_from_precomputed(nx_graphs[i], action_idx), device=self.dev
                    )
                    model.register_add_reactant_hook(self.add_reactant_hook)
                    add_reactant_logits = model.call_add_reactant_hook(
                        action_idx, self.graph_embeddings[i], self.graphs[i]
                    )
                    masked_logits = torch.zeros_like(add_reactant_logits) - 1000.0
                    masked_logits[masks.bool()] = add_reactant_logits[masks.bool()]
                    self.secondary_logits[0][i] = masked_logits

        logprobs = self.logsoftmax()

        # Initialize a tensor to hold the log probabilities for each action
        log_probs = torch.empty(len(actions), device=self.dev)
        for i, action in enumerate(actions):
            # Get the log probabilities for the current action type
            action_type, action_idx, action2_idx = action
            if self.ctx.aidx_to_action_type(action, fwd=self.fwd) == ActionType.Stop:
                log_prob = logprobs[action_type][i]
            elif self.ctx.aidx_to_action_type(action, fwd=self.fwd) == ActionType.ReactUni:
                log_prob = logprobs[action_type][i, action_idx]
            elif self.ctx.aidx_to_action_type(action, fwd=self.fwd) == ActionType.ReactBi:
                bireact_log_probs = logprobs[action_type]
                addreactant_log_probs = logprobs[self.action_type_to_logits_index[ActionType.AddReactant]]
                log_prob = bireact_log_probs[i, action_idx] + addreactant_log_probs[i, action2_idx]
            elif self.ctx.aidx_to_action_type(action, fwd=self.fwd) == ActionType.AddFirstReactant:
                log_prob = logprobs[self.action_type_to_logits_index[ActionType.AddFirstReactant]][i, action_idx]
            elif action == [0, None, None] and not self.fwd:
                log_prob = torch.tensor([0.0], device=self.dev, dtype=torch.float64)
            elif self.ctx.aidx_to_action_type(action, fwd=self.fwd) == ActionType.BckReactUni:
                log_prob = logprobs[self.action_type_to_logits_index[ActionType.BckReactUni]][i, action_idx]
            else:
                bireact_log_probs = logprobs[action_type]
                if action2_idx:  # if both products are BB and the remaining BB was selected randomly
                    log_prob = bireact_log_probs[i, action_idx] - math.log(2)
                else:
                    log_prob = bireact_log_probs[i, action_idx]
            log_probs[i] = log_prob
        return log_probs


def generate_backward_trajectory(g: Graph, traj_len: int) -> List[Tuple[Graph, int]]:
    """
    Generate a random trajectory that ends in g.

    Args:
        g (Graph): The target molecule.
        traj_len (int): The length of the trajectory.
    Returns:
        list: A list of tuples, where each tuple contains a molecule and an action.
    """
    raise NotImplementedError()


def generate_forward_trajectory(traj_len: int) -> List[Tuple[Graph, int]]:
    # Ideally use trajectories generated by AIZynthFinder - TODO
    # For now, generate a random trajectory that ends in g.
    """
    Generate a random trajectory.

    Args:
        traj_len (int): The length of the trajectory.
    Returns:
        list: A list of tuples, where each tuple contains a molecule and an action.
    """
    ctx = ReactionTemplateEnvContext()
    env = ReactionTemplateEnv()
    smi = random.choice(ctx.building_blocks)
    mol = Chem.MolFromSmiles(smi)
    fwd_traj = []
    for t in range(traj_len):
        masks = ctx.create_masks(mol, unimolecular=True)
        if sum(masks) != 0:
            # do unimolecular step
            p = [m / sum(masks) for m in masks]
            action = np.random.choice(ctx.num_unimolecular_rxns, p=p)
            fwd_traj.append((mol, (1, action, None)))
            mol = env.step(mol, (1, action, None))
            continue
        else:
            masks = ctx.create_masks(mol, unimolecular=False)
            if sum(masks) == 0:
                break
            p = [m / sum(masks) for m in masks]
            action = np.random.choice(ctx.num_bimolecular_rxns, p=p)
            reactant2_masks = ctx.create_masks_for_bb(mol, action)
            if sum(reactant2_masks) == 0:
                break
            p = [m / sum(reactant2_masks) for m in reactant2_masks]
            reactant2 = np.random.choice(ctx.num_building_blocks, p=p)
            fwd_traj.append((mol, (2, action, reactant2)))
            mol = env.step(mol, (2, action, reactant2))
    fwd_traj.append((mol, (0, None, None)))  # stop action
    return fwd_traj

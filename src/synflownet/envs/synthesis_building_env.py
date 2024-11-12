import csv
import math
import os
import pickle
import random
import warnings
from typing import Any, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import rdkit
import torch
import torch.nn as nn
import torch_geometric.data as gd
from rdkit import Chem, RDLogger
from rdkit.Chem import BondType, ChiralType, SaltRemover

from synflownet.envs.graph_building_env import ActionIndex, Graph, GraphAction, GraphActionType, GraphBuildingEnvContext
from synflownet.tasks.config import ReactionTaskConfig
from synflownet.utils.misc import get_worker_rng
from synflownet.utils.synthesis_utils import Reaction, get_mol_embeddings

logger = RDLogger.logger()
RDLogger.DisableLog("rdApp.*")

# Load templates and building blocks:
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
with open(os.path.join(repo_root, "data/building_blocks", ReactionTaskConfig.building_blocks_filename), "r") as file:
    BUILDING_BLOCKS = file.read().splitlines()
with open(os.path.join(repo_root, "data/templates", ReactionTaskConfig.templates_filename), "r") as file:
    TEMPLATES = file.read().splitlines()
if ReactionTaskConfig.reverse_templates_filename is not None:
    reverse_templates_filename = os.path.join(
        repo_root, "data/templates", ReactionTaskConfig.reverse_templates_filename
    )
    with open(reverse_templates_filename, "r") as file:
        REVERSE_TEMPLATES = file.read().splitlines()
else:
    REVERSE_TEMPLATES = None
with open(os.path.join(repo_root, "data/building_blocks", ReactionTaskConfig.precomputed_bb_masks_filename), "rb") as f:
    PRECOMPUTED_BB_MASKS = pickle.load(f)

if ReactionTaskConfig.building_blocks_costs is not None:
    with open(os.path.join(repo_root, "data/building_blocks", ReactionTaskConfig.building_blocks_costs), "r") as f:
        reader = csv.DictReader(f)
        # dictionary where key is index of row and value is the cost of the building block
        BUILDING_BLOCKS_COSTS = {i: float(row["score"]) for i, row in enumerate(reader)}
else:
    BUILDING_BLOCKS_COSTS = None

DEFAULT_CHIRAL_TYPES = [ChiralType.CHI_UNSPECIFIED, ChiralType.CHI_TETRAHEDRAL_CW, ChiralType.CHI_TETRAHEDRAL_CCW]

# Global variables are globally terrible. It's totally possible to refactor the code to avoid this one, but for now
# this very much does the job and avoids some annoying problems (i.e. pickling the context instance back and forth
# between processes)
_global_synth_ctx_instance = [None]


class ReactionTemplateEnvContext(GraphBuildingEnvContext):
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
            "Ni",
            "As",
            "Cd",
            "H",
            "Hg",
            "Mg",
            "Mn",
            "Se",
            "Pt",
            "Sb",
        ],
        chiral_types: List = DEFAULT_CHIRAL_TYPES,
        charges: List[int] = [-3, -2, -1, 0, 1, 2, 3],
        expl_H_range: List[int] = [0, 1, 2, 3, 4],  # for N
        allow_explicitly_aromatic: bool = True,
        allow_5_valence_nitrogen: bool = False,
        num_cond_dim: int = 0,
        reaction_templates: List[str] = TEMPLATES,
        rev_reaction_templates: Optional[List[str]] = REVERSE_TEMPLATES,
        building_blocks: List[str] = BUILDING_BLOCKS,
        precomputed_bb_masks: np.ndarray = PRECOMPUTED_BB_MASKS,
        fp_type: str = None,
        fp_path: str = None,
        add_hs: bool = False,  # NOTE: Needs to be true if using REAL reactions
        building_blocks_costs: Optional[dict] = BUILDING_BLOCKS_COSTS,
        strict_bck_masking: bool = True,
    ):
        """An env context for generating molecules by sequentially applying reaction templates.
        Contains functionalities to build molecular create masks for actions, and convert molecules to other representations.

        Args:
            atoms (list): List of atom symbols.
            chiral_types (list): List of chiral types.
            charges (list): List of charges.
            expl_H_range (list): List of explicit H counts.
            allow_explicitly_aromatic (bool): Whether to allow explicitly aromatic molecules.
            allow_5_valence_nitrogen (bool): Whether to allow N with valence of 5.
            num_cond_dim (int): The dimensionality of the observations' conditional information vector (if >0)
            reaction_templates (list): List of SMIRKS.
            rev_reaction_templates (list): List of reversed SMIRKS.
            building_blocks (list): List of SMILES strings of building blocks.
            precomputed_bb_masks (np.ndarray): Precomputed masks (for bimoelcular reactions) for building blocks and reaction templates.
            fp_type (str): The type of the molecules fingerprint to use.
            add_hs (bool): Whether to add hydrogens to the building blocks. NOTE: Needs to be true if using REAL reactions
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
        self.add_hs = add_hs
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
        self.strict_bck_masking = strict_bck_masking

        self.reactions = [Reaction(template=t) for t in reaction_templates]  # Reaction objects
        self.unimolecular_reactions = [r for r in self.reactions if r.num_reactants == 1]  # rdKit reaction objects
        self.bimolecular_reactions = [r for r in self.reactions if r.num_reactants == 2]
        self.num_unimolecular_rxns = len(self.unimolecular_reactions)
        self.num_bimolecular_rxns = len(self.bimolecular_reactions)

        if rev_reaction_templates is not None:
            self.reverse_reactions = [Reaction(template=t) for t in rev_reaction_templates]
            self.reverse_unimolecular_reactions = [r for r in self.reverse_reactions if r.num_products == 1]
            self.reverse_bimolecular_reactions = [r for r in self.reverse_reactions if r.num_products == 2]
        else:
            self.reverse_reactions = None

        self.building_blocks = building_blocks
        self.building_blocks_set = set(building_blocks)
        self.building_blocks_embs = get_mol_embeddings(building_blocks, fp_type=fp_type, fp_path=fp_path)
        self.building_blocks_dim = self.building_blocks_embs.shape[1]
        if ReactionTaskConfig.sanitize_building_blocks:
            print("Sanitizing building blocks ...")
            building_blocks_mols = [Chem.MolFromSmiles(bb) for bb in building_blocks]
            building_blocks_sanitized = []
            remover = (
                SaltRemover.SaltRemover()
            )  # some molecules are salts, we want the sanitized version of them not to have the salt
            for bb in building_blocks_mols:
                try:
                    bb = remover.StripMol(bb)
                    Chem.RemoveStereochemistry(bb)
                    building_blocks_sanitized.append(
                        Chem.MolToSmiles(self.graph_to_obj(self.obj_to_graph(bb)))
                    )  # graph_to_obj removes stereochemistry
                except Exception as e:
                    warnings.warn(f"Failed to sanitize building block {Chem.MolToSmiles(bb)}: {e}")
            self.building_blocks = building_blocks_sanitized

        if add_hs:
            self.building_blocks_mols = [Chem.AddHs(Chem.MolFromSmiles(bb)) for bb in self.building_blocks]
        else:
            self.building_blocks_mols = [Chem.MolFromSmiles(bb) for bb in self.building_blocks]

        self.num_building_blocks = len(building_blocks)
        self.precomputed_bb_masks = np.bool_(precomputed_bb_masks)
        self.precomputed_any_bb_reacts = self.precomputed_bb_masks.any(axis=2)
        self.bbs_costs = building_blocks_costs

        # Order in which models have to output logits
        self.action_type_order = [
            GraphActionType.Stop,
            GraphActionType.ReactUni,
            GraphActionType.ReactBi,
            GraphActionType.AddFirstReactant,
            GraphActionType.AddReactant,
        ]  # GraphActionType.AddReactant used separately in a hook during sampling
        self.bck_action_type_order = [
            GraphActionType.BckReactUni,
            GraphActionType.BckReactBi,
            GraphActionType.BckRemoveFirstReactant,
        ]

        # If there are no unimolecular reactions, remove the corresponding action type
        if self.num_unimolecular_rxns == 0:
            self.action_type_order.remove(GraphActionType.ReactUni)
            self.bck_action_type_order.remove(GraphActionType.BckReactUni)

        _global_synth_ctx_instance[0] = self

    def aidx_to_action_type(self, aidx: ActionIndex, fwd: bool = True):
        if fwd:
            action_type_order = self.action_type_order
        else:
            action_type_order = self.bck_action_type_order
        return action_type_order[aidx[0]]

    def action_type_to_aidx(self, action_type: GraphActionType, fwd: bool = True):
        if fwd:
            action_type_order = self.action_type_order
        else:
            action_type_order = self.bck_action_type_order
        return action_type_order.index(action_type)

    def ActionIndex_to_GraphAction(self, g: gd.Data, aidx: ActionIndex, fwd: bool = True) -> GraphAction:
        """Translate an ActionIndex to a GraphAction.

        Parameters
        ----------
        aidx: ActionIndex
            An integer representing an action.
        fwd: bool, default=True
            Whether the action is a forward or backward action.

        Returns
        -------
        action: GraphAction
            An action whose type is one of Stop, ReactUni, ReactBi, AddReactant, AddFirstReactant, BckReactUni, BckReactBi, BckRemoveFirstReactant.
        """
        if fwd:
            action_type_order = self.action_type_order
        else:
            action_type_order = self.bck_action_type_order

        bb = aidx.col_idx
        rxn = aidx.row_idx
        action_idx = aidx.action_type

        return GraphAction(action=action_type_order[action_idx], rxn=rxn, bb=bb)

    def GraphAction_to_ActionIndex(self, g: gd.Data, action: GraphAction, fwd: bool = True) -> int:
        """Translate a GraphAction to an ActionIndex.

        Parameters
        ----------
        action: GraphAction
            An action whose type is one of Stop, ReactUni, ReactBi, AddReactant, AddFirstReactant, BckReactUni, BckReactBi, BckRemoveFirstReactant.
        fwd: bool, default=True
            Whether the action is a forward or backward action.

        Returns
        -------
        action_idx: ActionIndex
            The ActionIndex corresponding to the action.
        """
        if fwd:
            action_type_order = self.action_type_order
        elif action.action is GraphActionType.Stop:
            return ActionIndex(action_type=0, row_idx=0, col_idx=0)
        else:
            action_type_order = self.bck_action_type_order

        type_idx = action_type_order.index(action.action)
        row_idx = action.rxn
        col_idx = action.bb

        return ActionIndex(action_type=type_idx, row_idx=row_idx, col_idx=col_idx)

    def create_masks(self, smi: Union[str, Chem.Mol, Graph], action_type: GraphActionType, traj_len: int) -> Union[List[int], Any]:  # type: ignore
        """Creates masks for actions given the molecule and action type.

        Args:
            mol (Chem.Mol): Molecule as a rdKit Mol object.
            action_type (GraphActionType): The type of action.
            traj_len (int): The length of the trajectory.
            fwd (bool): Whether it is a forward or a backward step.

        Returns:
            (List[int]): Masks for invalid actions.
        """
        # a dictionary where key is action type and value is the length of the masks array
        masks_len = {
            GraphActionType.Stop: 1,
            GraphActionType.ReactUni: self.num_unimolecular_rxns,
            GraphActionType.ReactBi: self.num_bimolecular_rxns,
            GraphActionType.AddReactant: self.num_building_blocks,
            GraphActionType.AddFirstReactant: self.num_building_blocks,
            GraphActionType.BckReactUni: self.num_unimolecular_rxns,
            GraphActionType.BckReactBi: self.num_bimolecular_rxns,
            GraphActionType.BckRemoveFirstReactant: 1,
        }

        if (
            traj_len == 0
            and action_type
            in [GraphActionType.ReactUni, GraphActionType.ReactBi, GraphActionType.AddReactant, GraphActionType.Stop]
        ) or (traj_len > 0 and action_type == GraphActionType.AddFirstReactant):
            return np.zeros(masks_len[action_type])
        elif traj_len == 0 and action_type == GraphActionType.AddFirstReactant:
            return np.ones(masks_len[action_type])  # no masks for AddFirstReactant
        elif traj_len == 1 and action_type == GraphActionType.Stop:
            return np.zeros(masks_len[action_type])
        elif traj_len != 1 and action_type == GraphActionType.Stop:
            return np.ones(masks_len[action_type])
        elif action_type == GraphActionType.BckRemoveFirstReactant:
            smi = Chem.MolToSmiles(self.get_mol(smi))  # type: ignore
            if smi in self.building_blocks_set:
                return np.ones(masks_len[action_type])
            else:
                return np.zeros(masks_len[action_type])
        else:
            mol = self.get_mol(smi)
            Chem.SanitizeMol(mol)  # type: ignore
            mol_copy = Chem.Mol(mol)  # type: ignore
            mol_copy = Chem.MolFromSmiles(Chem.MolToSmiles(mol_copy))  # type: ignore
            Chem.Kekulize(mol_copy, clearAromaticFlags=True)  # type: ignore
            if action_type == GraphActionType.ReactUni:
                reactions = self.unimolecular_reactions
            elif action_type == GraphActionType.BckReactUni:
                smi = Chem.MolToSmiles(self.get_mol(smi))  # type: ignore
                if self.reverse_reactions is not None:
                    reactions = self.reverse_unimolecular_reactions
                else:
                    reactions = self.unimolecular_reactions
            elif action_type == GraphActionType.BckReactBi:
                smi = Chem.MolToSmiles(self.get_mol(smi))  # type: ignore
                if self.reverse_reactions is not None:
                    reactions = self.reverse_bimolecular_reactions
                else:
                    reactions = self.bimolecular_reactions
            elif action_type in [GraphActionType.ReactBi]:
                reactions = self.bimolecular_reactions
            masks = np.ones(masks_len[action_type])
            r_possible_mask = self.check_if_any_bb_can_react(mol) if action_type == GraphActionType.ReactBi else None
            for idx, r in enumerate(reactions):
                if action_type is GraphActionType.ReactUni:
                    if not r.is_reactant(mol):
                        masks[idx] = 0
                elif action_type is GraphActionType.ReactBi:
                    if not r.is_reactant(mol):
                        masks[idx] = 0
                    else:
                        # check if for r and bb there is at least one other bb that can react in precomputed_bb_masks
                        masks[idx] = r_possible_mask[idx]
                else:
                    if self.reverse_reactions is not None:
                        if not (r.is_reactant(mol) or r.is_reactant(mol_copy)):
                            masks[idx] = 0
                    else:
                        if not (r.is_product(mol) or r.is_product(mol_copy)):
                            masks[idx] = 0

                    if self.strict_bck_masking:
                        if action_type == GraphActionType.BckReactBi:
                            # first we checked if the reactions matched, now we check if at least one of the products is a building block
                            if masks[idx] == 1:
                                ps = r.run_reverse_reactants((mol,))
                                ps_smiles = [Chem.MolToSmiles(p) for p in ps]  # type: ignore
                                for p_smiles in ps_smiles:
                                    if p_smiles in self.building_blocks_set:
                                        break
                                else:
                                    masks[idx] = 0
            return masks  # type: ignore

    def create_masks_for_bb(self, smi: Union[str, Chem.Mol, Graph], bimolecular_row_idx: int) -> List[bool]:  # type: ignore
        """Create masks for building blocks for a given molecule."""
        mol = self.get_mol(smi)
        Chem.SanitizeMol(mol)
        reaction = self.bimolecular_reactions[bimolecular_row_idx]
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
        self, smi: Union[str, Chem.Mol, Graph], bimolecular_row_idx: int
    ) -> List[bool]:
        """Creates masks for building blocks (for the 2nd reactant) for a given molecule and bimolecular reaction.
        Uses masks precomputed with data/building_blocks/precompute_bb_masks.py.

        Args:
            smi (Union[str, Chem.Mol, Graph]): Molecule as a rdKit Mol object.
            bimolecular_row_idx (int): Index of the bimolecular reaction.
        """
        mol = self.get_mol(smi)
        Chem.SanitizeMol(mol)
        reaction = self.bimolecular_reactions[bimolecular_row_idx]
        reactants = reaction.rxn.GetReactants()

        precomputed_bb_masks = self.precomputed_bb_masks[:, bimolecular_row_idx]
        match = np.array(
            [mol.HasSubstructMatch(reactants[1]), mol.HasSubstructMatch(reactants[0])]
        )
        return np.logical_or(*np.logical_and(precomputed_bb_masks, match[:, None])).astype(np.float64)

    def check_if_any_bb_can_react(self, smi: Union[str, Chem.Mol, Graph]) -> List[bool]:
        # Hmm, how would I write the above for all rows at once?
        # I want to return an array of bools, one for each reaction, that says whether any bb can react
        # self.precomputed_any_bb_reacts has shape (2, num_rxns)
        mol = self.get_mol(smi)
        all_matches = np.array([
            [mol.HasSubstructMatch(reactants[1]), mol.HasSubstructMatch(reactants[0])]
            for reaction in self.bimolecular_reactions
            for reactants in [reaction.rxn.GetReactants()]
        ]).T  # shape (2, num_rxns)
        return np.any(np.logical_and(self.precomputed_any_bb_reacts, all_matches), axis=0).astype(np.float64)

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
            mol = Chem.MolFromSmiles(smi, replacements={"[2H]": "[H]"})
            if self.add_hs:
                mol = Chem.AddHs(mol)
            return mol
        elif isinstance(smi, Chem.Mol):
            if self.add_hs:
                return Chem.AddHs(smi)
            else:
                return smi
        elif isinstance(smi, Graph):
            return self.graph_to_obj(smi)
        else:
            raise TypeError(f"{type(smi)} not supported, only `str` or `rdkit.Chem.Mol`")

    def obj_to_graph(self, mol: Chem.Mol) -> Graph:
        """Convert an RDMol to a Graph"""
        g = Graph()
        mol = Chem.Mol(mol)  # Make a copy
        try:
            mol.UpdatePropertyCache()
            if not self.allow_explicitly_aromatic:
                # If we disallow aromatic bonds, ask rdkit to Kekulize mol and remove aromatic bond flags
                Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception as e:
            warnings.warn(f"Failed to update property cache for mol {Chem.MolToSmiles(mol)}: {e}")
            # manually replace "c" with "C" in the smiles string
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol).replace("c", "C"))
            mol.UpdatePropertyCache()
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

    def graph_to_obj(self, g: Graph) -> Chem.Mol:
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
        try:
            Chem.SanitizeMol(mp)
        except Chem.AtomKekulizeException as e:
            rw_mol = Chem.RWMol(mp)
            for a in rw_mol.GetAtoms():
                if (not a.IsInRing()) and a.GetIsAromatic():
                    a.SetIsAromatic(False)
            for b in rw_mol.GetBonds():
                if (not b.IsInRing()) and b.GetIsAromatic():
                    b.SetIsAromatic(False)
            mp = Chem.Mol(rw_mol)
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

        # Mask the actions
        add_first_reactant_mask = self.create_masks(g, action_type=GraphActionType.AddFirstReactant, traj_len=traj_len)
        react_uni_mask = self.create_masks(g, action_type=GraphActionType.ReactUni, traj_len=traj_len)
        react_bi_mask = self.create_masks(g, action_type=GraphActionType.ReactBi, traj_len=traj_len)
        stop_mask = self.create_masks(g, action_type=GraphActionType.Stop, traj_len=traj_len)
        bck_react_uni_mask = self.create_masks(g, action_type=GraphActionType.BckReactUni, traj_len=traj_len)
        bck_react_bi_mask = self.create_masks(g, action_type=GraphActionType.BckReactBi, traj_len=traj_len)
        bck_remove_first_reactant_mask = self.create_masks(
            g, action_type=GraphActionType.BckRemoveFirstReactant, traj_len=traj_len
        )

        data = dict(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            # add attribute for masks
            add_first_reactant_mask=add_first_reactant_mask.reshape((1, len(add_first_reactant_mask))),
            react_uni_mask=react_uni_mask.reshape((1, len(react_uni_mask))),
            react_bi_mask=react_bi_mask.reshape((1, len(react_bi_mask))),
            stop_mask=stop_mask.reshape((1, len(stop_mask))),
            bck_react_uni_mask=bck_react_uni_mask.reshape((1, len(bck_react_uni_mask))),
            bck_react_bi_mask=bck_react_bi_mask.reshape((1, len(bck_react_bi_mask))),
            bck_remove_first_reactant_mask=bck_remove_first_reactant_mask.reshape(
                (1, len(bck_remove_first_reactant_mask))
            ),
        )
        # if react_uni/react_bi are masked, unmask the stop action
        if (np.sum(data["react_uni_mask"]) + np.sum(data["react_bi_mask"]) == 0) and traj_len > 0:
            data["stop_mask"] = np.ones_like(data["stop_mask"])
        data = gd.Data(**{k: torch.from_numpy(v) for k, v in data.items()})
        # for each masks type, print the action type and the length of the mask array
        data.edge_attr = data.edge_attr.to(torch.float32)
        data.x = data.x.to(torch.float32)
        return data

    def collate(self, graphs: List[gd.Data]):
        """Batch Data instances"""
        return gd.Batch.from_data_list(graphs, follow_batch=["edge_index"])

    def object_to_log_repr(self, g: Graph):
        """Convert a Graph to a string representation"""
        try:
            mol = self.graph_to_obj(g)
            assert mol is not None
            return Chem.MolToSmiles(mol)
        except Exception:
            return ""

    def traj_to_log_repr(self, traj: List[Tuple[Graph]]):
        """Convert a tuple of graph, action idx to a string representation, action idx"""
        smi_traj = []
        for i in traj:
            mol = self.graph_to_obj(i[0])
            assert mol is not None
            smi_traj.append((Chem.MolToSmiles(mol), i[1]))
        return str(smi_traj)

    def precompute_secondary_masks(self, actions, nx_graphs):
        idcs, masks = [], []
        for i, (g, a) in enumerate(zip(nx_graphs, actions)):
            if self.aidx_to_action_type(a) is GraphActionType.ReactBi:
                masks.append(self.create_masks_for_bb_from_precomputed(g, a.row_idx))
                idcs.append(i)
        return {GraphActionType.AddReactant: (torch.tensor(np.array(idcs)), torch.tensor(np.array(masks)))}

class ReactionTemplateEnv:
    """Molecules and reaction templates environment. The new, initial states are Enamine building block molecules.

    This environment specifies how to obtain new molecules from applying reaction templates to current molecules. Works by
    having the agent select a reaction template. Masks ensure that only valid templates are selected.
    """

    def __init__(
        self,
        reaction_templates: List[str] = TEMPLATES,
        building_blocks: List[str] = BUILDING_BLOCKS,
        precomputed_bb_masks: np.ndarray = PRECOMPUTED_BB_MASKS,
        fp_type: Optional[str] = None,
        fp_path: Optional[str] = None,
    ):
        """A reaction template environment instance"""
        self.ctx = ReactionTemplateEnvContext(
            reaction_templates=reaction_templates,
            building_blocks=building_blocks,
            precomputed_bb_masks=precomputed_bb_masks,
            fp_type=fp_type,
            fp_path=fp_path,
        )

    def new(self) -> Graph:
        smi = random.choice(self.ctx.building_blocks)
        mol = self.ctx.get_mol(smi)
        return self.ctx.obj_to_graph(mol)

    def empty_graph(self) -> Graph:
        return Graph()

    def step(self, smi: Union[str, Chem.Mol, Graph], action: GraphAction) -> Chem.Mol:
        """Applies the action to the current state and returns the next state.

        Args:
            mol (Chem.Mol): Current state as a SMILES string / RDKit mol / Graph.
            action (GraphAction): The action taken on the mol, indices must match

        Returns:
            (Chem.Mol): Next state as an RDKit mol.
        """
        mol = self.ctx.get_mol(smi)
        if action.action is GraphActionType.Stop:
            return mol
        elif action.action is GraphActionType.AddReactant or action.action is GraphActionType.AddFirstReactant:
            return self.ctx.get_mol(self.ctx.building_blocks[action.bb])
        elif action.action is GraphActionType.ReactUni:
            reaction = self.ctx.unimolecular_reactions[action.rxn]
            p = reaction.run_reactants((mol,))
            return p
        else:
            reaction = self.ctx.bimolecular_reactions[action.rxn]
            reactant2 = self.ctx.get_mol(self.ctx.building_blocks[action.bb])
            p = reaction.run_reactants((mol, reactant2))
            if p is None:
                raise ValueError(
                    f"Reaction {reaction.template} failed with reactants {Chem.MolToSmiles(mol)}, {Chem.MolToSmiles(reactant2)}"
                )
            return p

    def backward_step(self, smi: Union[str, Chem.Mol, Graph], action: GraphAction) -> Chem.Mol:
        """Applies the action to the current state and returns the previous (parent) state.

        Args:
            mol (Chem.Mol): Current state as an RDKit Mol object.
            action: Tuple[int, int]: Backward action indices to apply to the current state.
            (GraphActionType, reaction_template_idx)

        Returns:
            (Chem.Mol): Previous state as an RDKit mol and if the reaction is bimolecular,
            returns whether both products (reactants when fwd) are building blocks.
            This is important because if they are, we need to randomly select which to keep
            and this p_B of this action = 1/2.
            Smiles of building block that was not removed is also returned.
        """
        mol = self.ctx.get_mol(smi)
        # Chem.SanitizeMol(mol)
        if action.action is GraphActionType.BckRemoveFirstReactant:
            return self.ctx.get_mol(""), None, None
        elif action.action is GraphActionType.BckReactUni:
            if self.ctx.reverse_reactions is not None:
                reaction = self.ctx.reverse_unimolecular_reactions[action.rxn]
                return reaction.run_reverse_reactants((mol,), reaction.rxn), None, None
            else:
                reaction = self.ctx.unimolecular_reactions[action.rxn]
                return (
                    reaction.run_reverse_reactants((mol,)),
                    None,
                    None,
                )  # return the product and None (no reactant was removed)
        else:  # if bimolecular
            if self.ctx.reverse_reactions is not None:
                reaction = self.ctx.reverse_bimolecular_reactions[action.rxn]
                products = reaction.run_reverse_reactants((mol,), reaction.rxn)
            else:
                reaction = self.ctx.bimolecular_reactions[action.rxn]
                products = reaction.run_reverse_reactants((mol,))
            products_smi = [Chem.MolToSmiles(p) for p in products]

            both_are_bb = 0
            all_bbs = self.ctx.building_blocks
            if (products_smi[0] in all_bbs) and (products_smi[1] in all_bbs):
                both_are_bb = 1
                selected_product = random.choice(products)
                other_product = products_smi[0] if selected_product == products[1] else products_smi[1]
                other_product = self.ctx.building_blocks.index(other_product)  # get the index of the building block
            elif products_smi[0] in all_bbs:
                selected_product = products[1]
                other_product = self.ctx.building_blocks.index(products_smi[0])
            elif products_smi[1] in all_bbs:
                selected_product = products[0]
                other_product = self.ctx.building_blocks.index(products_smi[1])
            else:
                raise ValueError(
                    f"Neither product is a building block. Products: {products_smi}, reaction: {reaction.template}"
                )

            try:
                rw_mol = Chem.RWMol(selected_product)
                Chem.SanitizeMol(rw_mol)
                rw_mol = Chem.MolFromSmiles(Chem.MolToSmiles(rw_mol))
            # two types of error possible
            except (rdkit.Chem.rdchem.AtomKekulizeException, rdkit.Chem.rdchem.AtomValenceException) as e:
                if isinstance(e, Chem.AtomKekulizeException):
                    # non-ring atoms or bonds are marked aromatic
                    rw_mol = Chem.RWMol(selected_product)
                    for a in rw_mol.GetAtoms():
                        if (not a.IsInRing()) and a.GetIsAromatic():
                            a.SetIsAromatic(False)
                else:
                    # parse e to get the index of the atom that has the valence exception
                    for c in str(e):
                        # get the index of the atom that has the valence exception
                        if c.isdigit():
                            atom_idx = int(c)
                            break
                    rw_mol.GetAtomWithIdx(atom_idx).SetNumExplicitHs(0)
            Chem.SanitizeMol(rw_mol)
            rw_mol = Chem.MolFromSmiles(Chem.MolToSmiles(rw_mol))
            atoms_to_remove = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetSymbol() == "*"]
            for idx in sorted(
                atoms_to_remove, reverse=True
            ):  # Remove atoms in reverse order to avoid reindexing issues
                rw_mol.ReplaceAtom(idx, Chem.Atom("H"))
            atoms_to_remove = [
                atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetSymbol() in ["[CH]", "[C@@H]", "[C@H]"]
            ]
            for idx in sorted(
                atoms_to_remove, reverse=True
            ):  # Remove atoms in reverse order to avoid reindexing issues
                rw_mol.ReplaceAtom(idx, Chem.Atom("C"))
            try:
                rw_mol.UpdatePropertyCache()
            except Chem.rdchem.AtomValenceException as e:
                warnings.warn(f"{e}: Reaction {reaction.template}, product {Chem.MolToSmiles(selected_product)}")
            return rw_mol, both_are_bb, other_product

    def reverse(self, g: Graph, action: GraphAction):
        if action.action is GraphActionType.AddFirstReactant:
            return GraphAction(GraphActionType.BckRemoveFirstReactant)
        elif action.action is GraphActionType.ReactUni:
            return GraphAction(GraphActionType.BckReactUni, rxn=action.rxn)
        elif action.action is GraphActionType.ReactBi:
            bck_a = GraphAction(GraphActionType.BckReactBi, rxn=action.rxn, bb=0)
            _, both_are_bb, _ = self.backward_step(g, bck_a)
            if both_are_bb:
                return GraphAction(GraphActionType.BckReactBi, rxn=action.rxn, bb=1)
            else:
                return GraphAction(GraphActionType.BckReactBi, rxn=action.rxn, bb=0)
        elif action.action is GraphActionType.BckRemoveFirstReactant:
            return GraphAction(GraphActionType.AddFirstReactant)
        elif action.action is GraphActionType.BckReactUni:
            return GraphAction(GraphActionType.ReactUni, rxn=action.rxn)
        elif action.action is GraphActionType.BckReactBi:
            return GraphAction(GraphActionType.ReactBi, rxn=action.rxn, bb=action.bb)

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

    def parents_count(self, g: Graph) -> int:
        """Returns the number of parent molecules of a given molecule.

        Args:
            g (Graph): Molecule as a graph.

        Returns:
            (int): Number of parents.
        """
        parents_count = 0
        gd = self.ctx.graph_to_Data(g, traj_len=4)

        for _, atype in enumerate(self.ctx.bck_action_type_order):
            nza = getattr(gd, atype.mask_name)[0].nonzero()
            parents_count += len(nza)
        return parents_count

    def parents_count_idempotent(self, g: Graph) -> int:
        """Returns the number of parent molecules of a given molecule.

        Args:
            g (Graph): Molecule as a graph.

        Returns:
            (int): Number of parents.
        """
        parents = []
        parents_count = 0
        gd = self.ctx.graph_to_Data(g, traj_len=4)

        for tidx, atype in enumerate(self.ctx.bck_action_type_order):
            nza = getattr(gd, atype.mask_name)[0].nonzero()
            for o in nza:
                a = ActionIndex(tidx, o.item())  # o is idx of reaction
                ga = self.ctx.ActionIndex_to_GraphAction(gd, a, fwd=False)
                sp, both_are_bb, bb_idx = self.backward_step(g, ga)
                if both_are_bb:
                    for s_p in [sp, Chem.MolFromSmiles(self.ctx.building_blocks[bb_idx])]:
                        if s_p not in parents:
                            parents.append(s_p)
                            parents_count += 1
                else:
                    if sp not in parents:
                        parents.append(sp)
                        parents_count += 1
        return parents_count

    def count_backward_transitions(self, g: Graph, check_idempotent: bool = False) -> int:
        """Counts the number of backward transitions from a given molecule.

        Args:
            g (Graph): Molecule as a graph.

        Returns:
            (int): Number of possible backward transitions.
        """

        p_c = self.parents_count(g)
        if check_idempotent:
            p_c_2 = self.parents_count_idempotent(g)
            if p_c != p_c_2:
                print(p_c, p_c_2)
            return p_c_2
        return self.parents_count(g)


class ActionCategorical:
    def __init__(
        self,
        graphs: gd.Batch,
        graph_embeddings: torch.Tensor,
        raw_logits: List[torch.Tensor],
        keys: List[Union[str, None]],
        types: List[GraphActionType],
        action_masks: List[torch.Tensor] = None,
        fwd: bool = True,
    ):
        """A categorical distribution over the actions.

        Note on action-masking:
        Action masks depend on the environment logic (what are allowed v.s. prohibited actions).
        Thus, the action_masks should be created by the EnvContext (e.g. ReactionBuildingEnvContext)
        and passed to the ActionCategorical as a list of tensors. However, action masks
        should be applied to the logits within this class only to allow proper masking
        when computing log probabilities and sampling and avoid confusion about
        the state of the logits (masked or not) for external members.
        For this reason, the constructor takes as input the raw (unmasked) logits and the
        masks separately. The (masked) logits are cached in the _masked_logits attribute.
        Both the (masked) logits and the masks are private properties, and attempts to edit the masks or the logits will
        apply the masks to the raw_logits again.

        Parameters:
        graphs: Batch
            A batch of graphs to which the logits correspond.
        graph_embeddings: Tensor
            A tensor of shape (n, m) where n is the number of graphs and m is the embedding dimension.
        raw_logits: List[Tensor]
            A list of tensors representing unmasked logits, each of length (n,m) - n is number of graphs
            and there are m possible actions per action type.
            The length of the `logits` list is equal to the number of action
            types available.
        types: List[GraphActionType]
            The action type each logit group corresponds to.
        masks: List[Tensor], default=None
            If not None, a list of broadcastable tensors that multiplicatively
            mask out logits of invalid actions
        fwd: bool, default=True
            Whether the action space is for forward or backward actions.
        """
        self.graphs = graphs
        self.graph_embeddings = graph_embeddings
        self.dev = dev = graphs.x.device
        self.num_graphs = graphs.num_graphs
        if action_masks is not None:
            assert len(raw_logits) == len(action_masks)
        self.raw_logits = raw_logits
        self._action_masks = action_masks
        self.batch = [(torch.arange(graphs.num_graphs, device=dev)) for _ in range(len(raw_logits))]

        self.fwd = fwd
        self._epsilon = 1e-38
        self.logprobs = None
        self._has_secondary_masks = {}

        self.instantiate_action_hierarchy(types)
        self.instantiate_secondary_logits()
        self._apply_action_masks()

    def to(self, device):
        super().to(device)
        self.raw_logits = [logits.to(device) for logits in self.raw_logits]
        self._action_masks = [mask.to(device) for mask in self._action_masks]
        self._masked_logits = [logits.to(device) for logits in self._masked_logits]

    @property
    def ctx(self) -> ReactionTemplateEnvContext:
        # TODO: normally this is fine and the instance exists, but would be nice to do the usual singleton handling
        # more properly, especially within a multiprocessing context.
        return _global_synth_ctx_instance[0]

    @property
    def logits(self):
        return self._masked_logits

    @logits.setter
    def logits(self, new_raw_logits):
        self.raw_logits = new_raw_logits
        self._apply_action_masks()

    @property
    def action_masks(self):
        return self._action_masks

    @action_masks.setter
    def action_masks(self, new_action_masks):
        self._action_masks = new_action_masks
        self._apply_action_masks()

    def _apply_action_masks(self):
        self._masked_logits = (
            [self._mask(logits, mask) for logits, mask in zip(self.raw_logits, self._action_masks)]
            if self._action_masks is not None
            else self.raw_logits
        )

    def _mask(self, x, m):
        assert m.dtype in {torch.float32, torch.float64}
        return x.masked_fill(m == 0.0, -torch.inf)
        # return x.masked_fill(m == 0.0, -1000.0)

    def instantiate_action_hierarchy(self, types: List[GraphActionType]):
        """
        For fwd actions, there is a hierarchy of action types: AddFirstReactant, Stop, UniReact, BiReact to be sampled first, then AddReactant
        The logits are in the order: Stop, UniReact, BiReact, AddReactant
        """
        self.action_hierarchy = {
            "fwd": {
                "primary": types,
                "secondary": [GraphActionType.AddReactant],
            },
            "bck": {
                "primary": types,
            },
        }

        if self.fwd:
            self.action_type_to_logits_index = {
                action_type: i for i, action_type in enumerate(types + [GraphActionType.AddReactant])
            }
        else:
            self.action_type_to_logits_index = {action_type: i for i, action_type in enumerate(types)}

    def instantiate_secondary_logits(self):
        """
        The secondary logits are computed conditionally on a primary_action
        We thus instantiate them as placeholders before their associated masks and values are computed
        The reason we use placeholders and don't simply compute secondary_logits as local variable when needed
          is that there might be users of ActionCategorical that would loop over raw_logits, action_masks and batch indexes
          in order to compute some quantities or amend the masks/logits, and we want to make sure that the secondary_logits
          are accounted for in such cases. However, we don't want placeholders to accidentally be used in computations, hence
          setting them to nan.
        """
        key = "fwd" if self.fwd else "bck"
        if "secondary" in self.action_hierarchy[key]:
            secondary_action_types = self.action_hierarchy[key]["secondary"]
            secondary_action_idxs = [
                self.action_type_to_logits_index.get(action_type) for action_type in secondary_action_types
            ]

            assert len(secondary_action_idxs) == 1, "We only support one secondary_action for now"
            assert (
                self.ctx.action_type_to_aidx(GraphActionType.AddReactant) == secondary_action_idxs[0]
            ), "We only support AddReactant as a secondary action type for now"
            assert secondary_action_idxs[0] == len(self.raw_logits), "AddReactant should be the last action type"

        self.raw_logits.append(
            torch.full(size=(self.num_graphs, self.ctx.num_building_blocks), fill_value=torch.nan, device=self.dev)
        )
        self._action_masks.append(
            torch.ones(
                size=(self.num_graphs, self.ctx.num_building_blocks), dtype=self._action_masks[0].dtype, device=self.dev
            )
        )
        self.batch.append(torch.arange(self.graphs.num_graphs, device=self.dev))

    def set_secondary_masks(self, atype, idcs, masks):
        self._action_masks[self.ctx.action_type_to_aidx(atype)][idcs] = masks
        self._has_secondary_masks[atype] = True

    def get_logits_for_action_type(self, action_type):
        """Retrieve logits for a given action type."""
        index = self.action_type_to_logits_index.get(action_type)
        if index is not None:
            return self._masked_logits[index]
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
        primary_logits=None,
        secondary_logits=None,
        do_entropy: Optional[bool] = False,
    ):
        """Compute the max for each batch element in the batch of logits.

        Parameters
        ----------

        Returns
        -------
        overall_max_per_graph: Tensor
            A tensor of shape (n,m) where n is the number of graphs in the batch.
            Each element is the max value of the logits for the corresponding graph.
            m is 1 if there is one hierarchy of actions, and 2 if there are two hierarchies.
        """
        primary_logits = self.get_primary_logits() if primary_logits is None else primary_logits
        secondary_logits = self.get_secondary_logits() if secondary_logits is None else secondary_logits
        if do_entropy:
            primary_logits = [torch.where(l == -float("inf"), torch.tensor(-1000.0), l) for l in primary_logits]

        # Compute max for primary logits
        max_per_primary_type = [torch.max(tensor, dim=1)[0] for tensor in primary_logits]
        overall_max_per_graph_primary, _ = torch.max(torch.stack(max_per_primary_type), dim=0)

        # Compute max for secondary logits if they exist
        if secondary_logits:
            max_per_secondary_type = [torch.max(tensor, dim=1)[0] for tensor in secondary_logits]
            overall_max_per_graph_secondary, _ = torch.max(torch.stack(max_per_secondary_type), dim=0)
            assert not torch.isinf(
                overall_max_per_graph_secondary
            ).any(), "Inf values found in max secondary-logits. Might indicate that a forbidden action was selected."
        else:
            overall_max_per_graph_secondary = None

        return overall_max_per_graph_primary, overall_max_per_graph_secondary

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

    def logsoftmax(self, do_entropy: Optional[bool] = False):
        """Compute log-probabilities given logits"""
        if self.logprobs is not None:
            return self.logprobs
        # we need to compute the log-probabilities (1) for the primary logits and (2) for the secondary logits
        primary_logits = self.get_primary_logits()
        secondary_logits = self.get_secondary_logits()
        if do_entropy:
            primary_logits = [torch.where(l == -float("inf"), torch.tensor(-1000.0), l) for l in primary_logits]
            secondary_logits = [torch.where(l == -float("inf"), torch.tensor(-1000.0), l) for l in secondary_logits]
        max_logits_primary, max_logits_secondary = self._compute_batchwise_max(
            primary_logits=primary_logits,
            secondary_logits=secondary_logits,
            do_entropy=do_entropy
        )

        # substract max from logits for numerical stability
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

    def entropy(self, logprobs=None):
        """The entropy for each action categorical in the batch

        Parameters
        ----------
        logprobs: List[Tensor]
            The log-probablities of the policy (self.logsoftmax() by default)

        Returns
        -------
        entropies: Tensor
            The entropy of the policy for each action categorical in the batch
        """
        if logprobs is None:
            logprobs = self.logsoftmax(do_entropy=True)
        entropy = -sum([(i * i.exp()).sum(1) for i in logprobs])

        return entropy

    def get_addreactant_logits(self, model, rxn_id, emb, g):
        """Function to be called for the AddReactant action.
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
        rxn_features = torch.zeros(self.ctx.num_bimolecular_rxns).to(emb.device)
        rxn_features[rxn_id] = 1
        expanded_input = torch.cat((emb, rxn_features), dim=-1)
        mlp_outputs = model.call_mlp(GraphActionType.AddReactant.cname, expanded_input)
        return mlp_outputs

    def get_addreactant_logits_batched(self, model, rxn_ids, emb):
        """Function to be called for the AddReactant action.
        Parameters
        model : GraphTransformerReactionsGFN
            The model instance.
        rxn_ids : list[int]
            The IDs of the reactions selected by the sampler.
        emb : torch.Tensor
            The embedding tensor for the states.

        Returns
        torch.Tensor
            The logits or output of the MLP after being called with the expanded input.
        """
        # Convert `rxn_id` to a one-hot vector
        rxn_features = torch.nn.functional.one_hot(
            torch.tensor(rxn_ids), self.ctx.num_bimolecular_rxns
        ).to(emb.device)
        expanded_input = torch.cat((emb, rxn_features), dim=-1)
        mlp_outputs = model.call_mlp(GraphActionType.AddReactant.cname, expanded_input)
        return mlp_outputs

    def sample(
        self,
        nx_graphs: List[nx.Graph] = None,
        model: nn.Module = None,
        random_action_prob: float = 0.0,
        use_argmax: bool = False,
    ) -> List[ActionIndex]:
        """Samples from the categorical distribution"""
        primary_logits = self.get_primary_logits()
        device = primary_logits[0].device
        is_random_action = None

        if random_action_prob > 0.0:
            rng = get_worker_rng()
            # devices which graphs in the minibatch will get their action randomized
            is_random_action = torch.tensor(
                rng.uniform(size=self.graphs.num_graphs) < random_action_prob, device=device
            ).bool()
            # set the random action unmasked logits to a fixed value (e.g. 100) to ensure that the non-masked options are sampled uniformly
            primary_logits = [
                torch.where(
                    torch.logical_and(is_random_action[:, None], logits != -torch.inf),
                    torch.full_like(logits, 100.0),
                    logits,
                )
                for logits in primary_logits
            ]

        if use_argmax:
            argmax = self.argmax(x=primary_logits)
        else:
            u = [torch.rand(i.shape, device=device) for i in primary_logits]
            gumbel = [logit - (-noise.log()).log() for logit, noise in zip(primary_logits, u)]
            argmax = self.argmax(x=gumbel)

        bi_reacts = [self.ctx.aidx_to_action_type(t, fwd=self.fwd) == GraphActionType.ReactBi for t in argmax]
        if any(bi_reacts):
            all_add_reactant_logits = self.get_addreactant_logits_batched(
                        model, 
                        [t[1] for t, ok in zip(argmax, bi_reacts) if ok],
                        self.graph_embeddings[torch.tensor(bi_reacts, dtype=torch.bool, device=device)],
            )
        br_i = 0

        for i, t in enumerate(argmax):
            if self.ctx.aidx_to_action_type(t, fwd=self.fwd) == GraphActionType.AddFirstReactant:
                argmax[i] = ActionIndex(action_type=t[0], col_idx=t[1])

            elif self.ctx.aidx_to_action_type(t, fwd=self.fwd) == GraphActionType.Stop:
                argmax[i] = ActionIndex(action_type=t[0])

            elif self.ctx.aidx_to_action_type(t, fwd=self.fwd) in [
                GraphActionType.ReactUni,
                GraphActionType.BckReactUni,
            ]:
                argmax[i] = ActionIndex(action_type=t[0], row_idx=t[1])

            elif self.ctx.aidx_to_action_type(t, fwd=self.fwd) == GraphActionType.ReactBi:
                # we compute the AddReactant logits and masks conditionally to the sampled reaction
                add_reactant_masks = torch.tensor(
                    self.ctx.create_masks_for_bb_from_precomputed(nx_graphs[i], t[1]), device=device
                )
                # add_reactant_logits = self.get_addreactant_logits(model, t[1], self.graph_embeddings[i], self.graphs[i])
                add_reactant_logits = all_add_reactant_logits[br_i]
                br_i += 1
                masked_logits = self._mask(x=add_reactant_logits, m=add_reactant_masks)

                # apply random_action to secondary logits
                if is_random_action is not None:
                    masked_logits = torch.where(
                        torch.logical_and(is_random_action[i], masked_logits != -torch.inf),
                        torch.full_like(masked_logits, 100.0),
                        masked_logits,
                    )

                if use_argmax:
                    max_idx = int(masked_logits.argmax())
                else:
                    noise = torch.rand(masked_logits.shape, device=device)
                    gumbel = masked_logits - (-noise.log()).log()
                    max_idx = int(gumbel.argmax())
                assert add_reactant_masks[max_idx] == 1.0, "This index should not be masked"
                argmax[i] = ActionIndex(action_type=t[0], row_idx=t[1], col_idx=max_idx)

            elif self.ctx.aidx_to_action_type(t, fwd=self.fwd) == GraphActionType.BckRemoveFirstReactant:
                argmax[i] = ActionIndex(action_type=t[0])

            elif self.ctx.aidx_to_action_type(t, fwd=self.fwd) == GraphActionType.BckReactBi:
                argmax[i] = ActionIndex(action_type=t[0], row_idx=t[1])

            else:
                raise ValueError(f"Invalid action type: {self.ctx.aidx_to_action_type(t, fwd=self.fwd)}")

        return argmax

    def log_prob(
        self,
        actions: List[ActionIndex],
        nx_graphs: Optional[List[nx.Graph]] = None,
        model: Optional[nn.Module] = None,
        softmax: bool = True,
    ):
        """Access the log-probability of actions"""
        # Start by making sure all higher-order actions are accounted for
        if self.fwd:
            idcs_to_compute = []
            row_idcs = []
            for i, action in enumerate(actions):
                action_type, row_idx, col_idx = (
                    action.action_type,
                    action.row_idx,
                    action.col_idx,
                )
                if (
                    self.ctx.aidx_to_action_type(action, fwd=self.fwd)
                    == GraphActionType.ReactBi
                ):
                    # we compute the AddReactant logits and masks conditionally to the sampled reaction
                    if not self._has_secondary_masks.get(GraphActionType.AddReactant):
                        add_reactant_masks = torch.tensor(
                            self.ctx.create_masks_for_bb_from_precomputed(
                                nx_graphs[i], row_idx
                            ),
                            device=self.dev,
                        )
                        self._action_masks[
                            self.ctx.action_type_to_aidx(GraphActionType.AddReactant)
                        ][i] = add_reactant_masks
                    idcs_to_compute.append(i)
                    row_idcs.append(row_idx)
            if idcs_to_compute:
                idcs_to_compute = torch.tensor(idcs_to_compute, device=self.dev, dtype=torch.long)
                row_idcs = torch.tensor(row_idcs, device=self.dev, dtype=torch.long)
                add_reactant_logits = self.get_addreactant_logits_batched(
                    model, row_idcs, self.graph_embeddings[idcs_to_compute]
                )
                self.raw_logits[
                    self.ctx.action_type_to_aidx(GraphActionType.AddReactant)
                ][idcs_to_compute] = add_reactant_logits
                self._apply_action_masks()

        logprobs = self.logsoftmax() if softmax else self.get_primary_logits() + self.get_secondary_logits()

        # Initialize a tensor to hold the log probabilities for each action
        log_probs = torch.empty(len(actions), device=self.dev)
        for i, action in enumerate(actions):
            # Get the log probabilities for the current action type
            action_type, row_idx, col_idx = action.action_type, action.row_idx, action.col_idx
            if self.ctx.aidx_to_action_type(action, fwd=self.fwd) == GraphActionType.Stop:
                log_prob = logprobs[action_type][i]
            elif self.ctx.aidx_to_action_type(action, fwd=self.fwd) == GraphActionType.ReactUni:
                log_prob = logprobs[action_type][i, row_idx]
            elif self.ctx.aidx_to_action_type(action, fwd=self.fwd) == GraphActionType.ReactBi:
                bireact_log_probs = logprobs[action_type]
                addreactant_log_probs = logprobs[self.action_type_to_logits_index[GraphActionType.AddReactant]]
                log_prob = bireact_log_probs[i, row_idx] + addreactant_log_probs[i, col_idx]
            elif self.ctx.aidx_to_action_type(action, fwd=self.fwd) == GraphActionType.AddFirstReactant:
                log_prob = logprobs[self.action_type_to_logits_index[GraphActionType.AddFirstReactant]][i, col_idx]

            elif self.ctx.aidx_to_action_type(action, fwd=True) == GraphActionType.Stop and not self.fwd:
                log_prob = torch.tensor([0.0], device=self.dev, dtype=torch.float64)
            elif self.ctx.aidx_to_action_type(action, fwd=self.fwd) == GraphActionType.BckReactUni:
                log_prob = logprobs[self.action_type_to_logits_index[GraphActionType.BckReactUni]][i, row_idx]
            elif self.ctx.aidx_to_action_type(action, fwd=self.fwd) == GraphActionType.BckRemoveFirstReactant:
                log_prob = logprobs[self.action_type_to_logits_index[GraphActionType.BckRemoveFirstReactant]][i]
            else:
                bireact_log_probs = logprobs[action_type]
                if col_idx:  # if both products are BB and the remaining BB was selected randomly
                    log_prob = bireact_log_probs[i, row_idx] - math.log(2)
                else:
                    log_prob = bireact_log_probs[i, row_idx]
            log_probs[i] = log_prob
        return log_probs

    def logsumexp(
        self,
        actions: List[ActionIndex],
        nx_graphs: List[nx.Graph],
        model: nn.Module,
        logits: Optional[List[torch.Tensor]] = None,
    ):
        """Reduces `x` (the logits by default) to one scalar per graph"""

        # make sure secondary logits are masked correctly
        if self.fwd:
            idcs_to_compute = []
            row_idcs = []
            for i, action in enumerate(actions):
                action_type, row_idx, col_idx = (
                    action.action_type,
                    action.row_idx,
                    action.col_idx,
                )
                if (
                    self.ctx.aidx_to_action_type(action, fwd=self.fwd)
                    == GraphActionType.ReactBi
                ):
                    # we compute the AddReactant logits and masks conditionally to the sampled reaction
                    if not self._has_secondary_masks.get(GraphActionType.AddReactant):
                        add_reactant_masks = torch.tensor(
                            self.ctx.create_masks_for_bb_from_precomputed(
                                nx_graphs[i], row_idx
                            ),
                            device=self.dev,
                        )
                        self._action_masks[
                            self.ctx.action_type_to_aidx(GraphActionType.AddReactant)
                        ][i] = add_reactant_masks
                    idcs_to_compute.append(i)
                    row_idcs.append(row_idx)
            if idcs_to_compute:
                idcs_to_compute = torch.tensor(idcs_to_compute, device=self.dev, dtype=torch.long)
                row_idcs = torch.tensor(row_idcs, device=self.dev, dtype=torch.long)
                add_reactant_logits = self.get_addreactant_logits_batched(
                    model, row_idcs, self.graph_embeddings[idcs_to_compute]
                )
                self.raw_logits[
                    self.ctx.action_type_to_aidx(GraphActionType.AddReactant)
                ][idcs_to_compute] = add_reactant_logits
                self._apply_action_masks()

        if logits is None:
            primary_logits = self.get_primary_logits()
            secondary_logits = self.get_secondary_logits()

        else:
            primary_logits = [
                logits[self.action_type_to_logits_index[t]] for t in self.action_hierarchy["fwd"]["primary"]
            ]
            secondary_logits = [
                logits[self.action_type_to_logits_index[t]] for t in self.action_hierarchy["fwd"]["secondary"]
            ]

        max_logits_primary, max_logits_secondary = self._compute_batchwise_max(primary_logits, secondary_logits)

        # substract max from logits for numerical stability
        corr_logits_primary = [tensor - max_logits_primary.view(-1, 1) for tensor in primary_logits]
        exp_logits_primary = [i.exp().clamp(self._epsilon) for i in corr_logits_primary]
        # compute logZ for primary logits
        merged_exp_logits_primary = torch.cat(exp_logits_primary, dim=1)
        logZ_primary = merged_exp_logits_primary.sum(dim=1).log()
        # re-adding the substracted max back to the logsumexp
        logsumexp = logZ_primary + max_logits_primary

        # if there are secondary logits, compute logsumexp for them
        if max_logits_secondary is not None:
            corr_logits_secondary = [tensor - max_logits_secondary.view(-1, 1) for tensor in secondary_logits]
            exp_logits_secondary = [i.exp().clamp(self._epsilon) for i in corr_logits_secondary]
            merged_exp_logits_secondary = torch.cat(exp_logits_secondary, dim=1)
            logZ_secondary = merged_exp_logits_secondary.sum(dim=1).log()
            logsumexp2 = logZ_secondary + max_logits_secondary

        for i, action in enumerate(actions):
            # p_ij = sum(e^i*e^j)/(Z_1*Z_2); log(p_ij) = log(sum(e^i*e^j)) - log(Z_1) - log(Z_2)
            # overall partition function per graph
            # only add the secondary logsumexp if the action for that graph is ReactBi
            if self.ctx.aidx_to_action_type(action, fwd=self.fwd) == GraphActionType.ReactBi:
                # logsumexp[i] += logsumexp2[i]
                logsumexp[i] = torch.logaddexp(logsumexp[i], logsumexp2[i])

        return logsumexp


def generate_forward_trajectory(traj_len: int, ctx: None, env: None) -> List[Tuple[Graph, int]]:
    # Generate a random trajectory starting from a random building block.
    """
    Generate a random trajectory.

    Args:
        traj_len (int): The length of the trajectory.
    Returns:
        list: A list of tuples, where each tuple contains a molecule and an action.
    """
    if ctx is None:
        ctx = ReactionTemplateEnvContext()
        env = ReactionTemplateEnv()

    fwd_traj = []
    g = env.empty_graph()
    for t in range(traj_len):
        gd = ctx.graph_to_Data(g, traj_len=t)
        nza = []
        action_type_order = [a for a in ctx.action_type_order if a != GraphActionType.AddReactant]
        for tidx, atype in enumerate(action_type_order):
            nza.append(getattr(gd, atype.mask_name)[0].nonzero().flatten().tolist())
        actions = [(random.choice(nza[i]), i) for i in range(len(nza)) if len(nza[i]) > 0]
        # choose a random action and get the index of the list it's part of
        action, action_idx = random.choice(actions)
        a = GraphAction(action_type_order[action_idx], rxn=action)
        if a.action == GraphActionType.ReactBi:
            reactant_masks = ctx.create_masks_for_bb_from_precomputed(g, a.rxn)
            bb_idx = random.choice([i for i, v in enumerate(reactant_masks) if v != 0])
            a.bb = bb_idx
        m = env.step(g, a)
        fwd_traj.append((g, a))
        if a.action == GraphActionType.Stop:
            break
        g = ctx.obj_to_graph(m)

    return fwd_traj


def generate_backward_trajectory(traj_len: int, s: str, ctx: None, env: None):
    # Generate a random trajectory starting from a random building block.
    """
    Generate a random bck trajectory.
    """
    if ctx is None:
        ctx = ReactionTemplateEnvContext()
        env = ReactionTemplateEnv()

    bck_traj = []
    g = ctx.obj_to_graph(Chem.MolFromSmiles(s))
    for t in range(traj_len):
        gd = ctx.graph_to_Data(g, traj_len=t)
        nza = []
        action_type_order = [a for a in ctx.bck_action_type_order]
        for tidx, atype in enumerate(action_type_order):
            nza.append(getattr(gd, atype.mask_name)[0].nonzero().flatten().tolist())
        try:
            actions = [(random.choice(nza[i]), i) for i in range(len(nza)) if len(nza[i]) > 0]
            # choose a random action and get the index of the list it's part of
            action, action_idx = random.choice(actions)
            a = GraphAction(action_type_order[action_idx], rxn=action)
            m, both_are_bb, bb_idx = env.backward_step(g, a)
            bck_traj.append((g, a))
            g = ctx.obj_to_graph(m)
        except Exception:
            continue

    return bck_traj

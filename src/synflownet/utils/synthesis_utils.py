import os
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, SaltRemover, rdChemReactions

from synflownet.tasks.config import ReactionTaskConfig

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)

if os.path.exists(os.path.join(repo_root, "data/building_blocks", ReactionTaskConfig.building_blocks_filename)):
    with open(os.path.join(repo_root, "data/building_blocks", ReactionTaskConfig.building_blocks_filename), "r") as file:
        BUILDING_BLOCKS = file.read().splitlines()
else: 
    BUILDING_BLOCKS = None

building_blocks = BUILDING_BLOCKS




if building_blocks is not None: # hotfix to avoid circular dependency on file

    if ReactionTaskConfig.sanitize_building_blocks:
        building_blocks_sanitized = []
        building_blocks_mols = [Chem.MolFromSmiles(bb) for bb in building_blocks]
        remover = (
            SaltRemover.SaltRemover()
        )  # some molecules are salts, we want the sanitized version of them not to have the salt
        for bb in building_blocks_mols:
            bb = remover.StripMol(bb)
            Chem.RemoveStereochemistry(bb)
            building_blocks_sanitized.append(Chem.MolToSmiles(bb))
        building_blocks_sanitized = set(building_blocks_sanitized)
    else:
        building_blocks_sanitized = set(building_blocks)


def run_reaction_with_fallback(rxn, product):
    """Run reaction and fallback to kekulized product if necessary."""
    rs = rxn.RunReactants(product)
    if len(rs) == 0:
        product = Chem.MolFromSmiles(Chem.MolToSmiles(product[0]))
        Chem.Kekulize(product, clearAromaticFlags=True)
        rs = rxn.RunReactants((product,))
    if len(rs) == 0:
        raise ValueError(f"Reaction did not yield any products. Product: {Chem.MolToSmiles(product)}")
    return rs


def clean_smiles(smi):
    replacements = {
        "~": "-",
        "[C]": "C",
        "[CH]": "C",
        "[C@@H]": "C",
        "[C@H]": "C",
        "[nH2+]": "N",
    }
    for old, new in replacements.items():
        smi = smi.replace(old, new)
    return smi


def clean_molecule(mol):
    if mol is None:
        return None
    try:
        smi = Chem.MolToSmiles(mol)
    except Exception:
        return mol  # Return the original molecule if SMILES conversion fails
    smi_clean = clean_smiles(smi)
    mol_clean = Chem.MolFromSmiles(smi_clean)
    return mol_clean if mol_clean is not None else mol


def try_sanitize_molecule(mol):
    try:
        Chem.SanitizeMol(mol)
    except (Chem.rdchem.AtomKekulizeException, Chem.rdchem.AtomValenceException, Chem.rdchem.KekulizeException):
        pass


class Reaction:
    def __init__(self, template=None):
        self.template = template
        self.rxn = self.__init_reaction()
        self.num_reactants = self.rxn.GetNumReactantTemplates()
        self.num_products = self.rxn.GetNumProductTemplates()
        # Extract reactants, agents, products
        reactants, agents, products = self.template.split(">")
        if self.num_reactants == 1:
            self.reactant_template = list((reactants,))
        else:
            self.reactant_template = list(reactants.split("."))
        self.product_template = products

    def __init_reaction(self) -> Chem.rdChemReactions.ChemicalReaction:
        """Initializes a reaction by converting the SMARTS-pattern to an `rdkit` object."""
        rxn = AllChem.ReactionFromSmarts(self.template)
        rdChemReactions.ChemicalReaction.Initialize(rxn)
        return rxn

    def reverse_template(self):
        """Reverses a reaction template and returns an initialized, reversed reaction object."""
        rxn = AllChem.ChemicalReaction()
        for i in range(self.rxn.GetNumReactantTemplates()):
            rxn.AddProductTemplate(self.rxn.GetReactantTemplate(i))
        for i in range(self.rxn.GetNumProductTemplates()):
            rxn.AddReactantTemplate(self.rxn.GetProductTemplate(i))
        rxn.Initialize()
        return rxn

    def is_reactant(self, mol: Chem.Mol, rxn: rdChemReactions = None) -> bool:
        """Checks if a molecule is a reactant for the reaction."""
        if rxn is None:
            rxn = self.rxn
        return self.rxn.IsMoleculeReactant(mol)

    def is_product(self, mol: Chem.Mol, rxn: rdChemReactions = None) -> bool:
        """Checks if a molecule is a reactant for the reaction."""
        if rxn is None:
            rxn = self.rxn
        return self.rxn.IsMoleculeProduct(mol)

    def is_reactant_first(self, mol: Chem.Mol) -> bool:
        """Checks if a molecule is the first reactant for the reaction."""
        pattern = Chem.MolFromSmarts(self.reactant_template[0])
        return mol.HasSubstructMatch(pattern)

    def is_reactant_second(self, mol: Chem.Mol) -> bool:
        """Checks if a molecule is the second reactant for the reaction."""
        pattern = Chem.MolFromSmarts(self.reactant_template[1])
        return mol.HasSubstructMatch(pattern)

    def run_reactants(
        self, reactants: Tuple[Union[Chem.Mol, str, None]], rxn: rdChemReactions = None, keep_main: bool = True
    ) -> Union[Chem.Mol, None]:
        """Runs the reaction on a set of reactants and returns the product.

        Args:
            reactants: A tuple of reactants to run the reaction on.
            keep_main: Whether to return the main product or all products. Default is True.

        Returns:
            The product of the reaction or `None` if the reaction is not possible.
        """
        if len(reactants) not in (1, 2):
            raise ValueError(f"Can only run reactions with 1 or 2 reactants, not {len(reactants)}.")
        if rxn is None:
            rxn = self.rxn
        if self.num_reactants == 1:
            if len(reactants) == 2:  # Provided two reactants for unimolecular reaction; not possible
                return None
            if not self.is_reactant(reactants[0]):
                return None
        elif self.num_reactants == 2:
            if self.is_reactant_first(reactants[0]) and self.is_reactant_second(reactants[1]):
                pass
            elif self.is_reactant_first(reactants[1]) and self.is_reactant_second(reactants[0]):
                reactants = tuple(reversed(reactants))
            else:
                return None
        else:
            raise ValueError("Reaction is neither unimolecular nor bimolecular.")

        # Run reaction
        ps = rxn.RunReactants(reactants)

        if len(ps) == 0:
            raise ValueError("Reaction did not yield any products.")
        else:
            for i in range(len(ps)):
                # return the first product that when canonicalized does not return None
                p = ps[i][0]
                p_canon = Chem.MolFromSmiles(Chem.MolToSmiles(p))
                if p_canon is not None:
                    try:
                        Chem.SanitizeMol(p_canon)
                    except (Chem.rdchem.KekulizeException, Chem.rdchem.AtomValenceException) as e:
                        warnings.warn(
                            f"{e}: Sanitization exception with reaction {self.template} and reactants {Chem.MolToSmiles(reactants[0])}, {Chem.MolToSmiles(reactants[1])} in the fwd step."
                        )
                    p = Chem.RemoveHs(p_canon)
                    return p
        return ps[0][0]

    def run_reverse_reactants(
        self, product: Tuple[Chem.Mol], rxn: rdChemReactions = None, keep_main: bool = True
    ) -> Union[Chem.Mol, None]:
        """Runs the reverse reaction on a product, to return the reactants.

        Args:
            product: A tuple of Chem.Mol object of the product (now reactant) to run the reverse reaction on.
            keep_main: Whether to return the main product or all products. Default is True.

        Returns:
            The product (reactant(s)) of the reaction or `None` if the reaction is not possible.
        """
        if rxn is None:
            rxn = self.reverse_template()
            self.num_products = self.num_reactants

        rs = run_reaction_with_fallback(rxn, product)

        if self.num_products == 1:
            return rs[0][0] if keep_main else rs
        elif self.num_products == 2:
            if len(rs) == 1:
                reactant_list = list(rs[0])
                for mol in reactant_list:
                    try_sanitize_molecule(mol)
                rs_canon = [Chem.MolFromSmiles(Chem.MolToSmiles(mol)) for mol in reactant_list]
                if all(mol is not None for mol in rs_canon):
                    clean_list = [clean_molecule(mol) for mol in reactant_list]
                    return clean_list
                else:
                    clean_list = [clean_molecule(mol) for mol in reactant_list]
                    return clean_list
            else:
                for i, reactant_set in enumerate(rs):
                    reactant_list = list(reactant_set)
                    reactant_list = [clean_molecule(mol) if mol is not None else None for mol in reactant_list]
                    for mol in reactant_list:
                        try_sanitize_molecule(mol)
                    rs_canon = [Chem.MolFromSmiles(Chem.MolToSmiles(mol)) for mol in reactant_list]  # canonicalize
                    if all(mol is not None for mol in rs_canon):
                        reactant_smiles = [Chem.MolToSmiles(mol) for mol in rs_canon]
                        if any(smi in building_blocks_sanitized for smi in reactant_smiles):
                            return rs_canon
                        for r_i in reactant_list[2:]:
                            if Chem.MolToSmiles(r_i) in building_blocks_sanitized:
                                return reactant_list
                # if no reactants are building blocks, return the first reactants
                if Chem.MolFromSmiles(Chem.MolToSmiles(rs[0][0])) is not None:
                    rs = list(rs[0])
                else:
                    rs = list(rs[1])
                for mol in rs:
                    try_sanitize_molecule(mol)
                rs_canon = [Chem.MolFromSmiles(Chem.MolToSmiles(r)) for r in rs]
                if all(r is not None for r in rs_canon):
                    clean_list = [clean_molecule(mol) for mol in rs_canon]
                    return clean_list
                else:
                    clean_list = [clean_molecule(mol) for mol in rs]
                    return clean_list
                raise ValueError("Reaction did not yield any products.")
        else:
            raise ValueError("Reaction is neither unimolecular nor bimolecular.")


def get_mol_embeddings(smiles: List[str], fp_type: Optional[str], fp_path: Optional[str]) -> torch.Tensor:
    if fp_type is None and fp_path is None:
        print("No fingerprint type or path provided. Setting BB embeddings to empty tensor.")
        return torch.empty((len(smiles), 0))
    if fp_type == "molgps":
        assert fp_path is not None, "Path to precomputed MolGPS embeddings must be provided."
        molgps = torch.load(fp_path)
        smi2emb = dict(zip(molgps["smiles"], molgps["fps_mpnn"]))
        embeddings = torch.stack([smi2emb[smi] for smi in smiles], dim=0)
        return embeddings
    if fp_type.startswith("morgan"):
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        embeddings = torch.stack([mol2fingerprint(mol, fp_type=fp_type) for mol in mols], dim=0)
        return embeddings
    raise NotImplementedError(f"Fingerprint {fp_type} not implemented.")


def mol2fingerprint(rdmol: Chem.Mol, fp_type: str = "morgan_1024"):
    """Converts an RDKit molecule to a fingerprint."""
    if fp_type.startswith("morgan"):
        nbits = int(fp_type.split("_")[1])
        fp = AllChem.GetMorganFingerprintAsBitVect(rdmol, 2, nBits=nbits)
        fingerprint = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, fingerprint)
        fingerprint = torch.tensor(fingerprint, dtype=torch.float32)
    else:
        raise NotImplementedError(f"Fingerprint {fp_type} not implemented.")

    return fingerprint

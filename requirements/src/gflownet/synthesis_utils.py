import warnings
from typing import Tuple, Union

from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions


class Reaction:
    def __init__(self, template=None):
        self.template = template
        self.rxn = self.__init_reaction()
        self.num_reactants = self.rxn.GetNumReactantTemplates()
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

        if keep_main:
            p = ps[0][0]
            try:
                Chem.SanitizeMol(p)
            except (Chem.rdchem.KekulizeException, Chem.rdchem.AtomValenceException) as e:
                warnings.warn(
                    f"{e}: Reaction {self.template}, reactants {Chem.MolToSmiles(reactants[0])}, {Chem.MolToSmiles(reactants[1])}"
                )
            p = Chem.RemoveHs(p)
            return p
        else:
            return ps

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

        # Run reaction
        if self.num_reactants == 1:
            rs = rxn.RunReactants(product)
            if keep_main:
                return rs[0][0]
            else:
                return rs
        elif self.num_reactants == 2:
            try:
                rs = list(rxn.RunReactants(product)[0])
            except IndexError:  # Because reaction SMARTS and product can be either kekulized or not
                product = Chem.MolFromSmiles(Chem.MolToSmiles(product[0]))
                Chem.Kekulize(product, clearAromaticFlags=True)
                rs = list(rxn.RunReactants((product,))[0])
            reactants_smi = [Chem.MolToSmiles(r) for r in rs]
            for i, s in enumerate(reactants_smi):
                if "[CH]" in s:
                    s = s.replace("[CH]", "C")
                    rs[i] = Chem.MolFromSmiles(s)
            if len(rs) == 0:
                raise ValueError("Reaction did not yield any products.")
            return rs
        else:
            raise ValueError("Reaction is neither unimolecular nor bimolecular.")

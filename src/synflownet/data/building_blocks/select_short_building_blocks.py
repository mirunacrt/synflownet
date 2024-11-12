# Determine how many of the Enamine building blocks match the templates we have.

import argparse
from pathlib import Path

from rdkit import Chem, RDLogger

logger = RDLogger.logger()
RDLogger.DisableLog("rdApp.*")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select building blocks with numatoms < num_atoms")
    parser.add_argument("--filename", type=str, help="Path to input building blocks", default="building_blocks.txt")
    parser.add_argument(
        "--output_filename", type=str, help="Path to output building blocks", default="short_building_blocks.txt"
    )
    parser.add_argument("--num_atoms", type=int, help="Maximum number of atoms", default=20)
    args = parser.parse_args()

    with open(Path(__file__).parent / args.filename, "r") as file:
        bb_list = file.read().splitlines()

    short_bbs = []
    for smiles in bb_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol.GetNumAtoms() < args.num_atoms:
            short_bbs.append(smiles)

    new_filename = args.output_filename
    with open(Path(__file__).parent / new_filename, "w") as file:
        for bb in short_bbs:
            file.write(bb + "\n")

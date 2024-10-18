import argparse
import warnings

from rdkit import Chem
from rdkit.Chem import SaltRemover

from gflownet.envs.synthesis_building_env import ReactionTemplateEnvContext


def sanitize_building_blocks(building_blocks: list[str]) -> list[str]:
    ctx = ReactionTemplateEnvContext()

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
                Chem.MolToSmiles(ctx.graph_to_obj(ctx.obj_to_graph(bb)))
            )  # graph_to_obj removes stereochemistry
        except Exception as e:
            warnings.warn(f"Failed to sanitize building block {Chem.MolToSmiles(bb)}: {e}")
    return building_blocks_sanitized


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanitize building blocks")
    parser.add_argument("--building_blocks_filename", type=str, help="Path to building blocks")
    parser.add_argument("--output_filename", type=str, help="Path to output sanitized building blocks")
    args = parser.parse_args()

    with open(args.building_blocks_filename, "r") as file:
        building_blocks = file.read().splitlines()

    print("Sanitizing building blocks ...")
    building_blocks_sanitized = sanitize_building_blocks(building_blocks)
    # remove empty strings
    building_blocks_sanitized = [bb for bb in building_blocks_sanitized if bb]

    new_filename = args.output_filename
    with open(new_filename, "w") as file:
        for bb in building_blocks_sanitized:
            file.write(bb + "\n")

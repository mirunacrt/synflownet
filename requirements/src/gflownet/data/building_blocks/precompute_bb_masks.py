import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from gflownet.envs.synthesis_building_env import ReactionTemplateEnvContext

PATH = Path(__file__).parent / "precomputed_bb_masks.pkl"


def precompute_bb_masks():
    ctx = ReactionTemplateEnvContext()

    print("Precomputing building blocks masks for each reaction and reactant position...")
    masks = np.zeros((2, ctx.num_bimolecular_rxns, ctx.num_building_blocks))
    for rxn_i in tqdm(range(ctx.num_bimolecular_rxns)):
        reaction = ctx.bimolecular_reactions[rxn_i]
        reactants = reaction.rxn.GetReactants()
        for bb_j, bb in enumerate(ctx.building_blocks_mols):
            if bb.HasSubstructMatch(reactants[0]):
                masks[0, rxn_i, bb_j] = 1
            if bb.HasSubstructMatch(reactants[1]):
                masks[1, rxn_i, bb_j] = 1

    print(f"Saving precomputed masks to of shape={masks.shape} to {PATH}")
    with open(PATH, "wb") as f:
        pickle.dump(masks, f)


if __name__ == "__main__":
    precompute_bb_masks()
    print("Done!")
    with open(PATH, "rb") as f:
        masks = pickle.load(f)
    print(masks.shape)

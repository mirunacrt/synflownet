import ast
import os
import re
import sqlite3
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.ML.Cluster import Butina
from scipy.spatial.distance import pdist

from synflownet.envs.graph_building_env import GraphAction, GraphActionType


def convert_to_strings_and_ints(input_str):
    tuples_list = ast.literal_eval(input_str)
    results = []
    for item in tuples_list:
        result_string = item[0]
        result_int = item[1]
        results.append((result_string, result_int))
    return results


def parse_graph_action_list(s):
    # Define the regular expression pattern to match the list of tuples string
    # tuple_pattern = r"\('([^']*)', <GraphActionType\.([^,]+), >\)"
    tuple_pattern = r"\('([^']*)', <GraphActionType\.([^,]+), (\d+), (\d+)>"
    matches = re.findall(tuple_pattern, s)
    tuples_list = []
    if matches:
        tuples_list = []
        for item in matches:
            molecule = item[0]
            action_type_str = item[1]
            rxn = int(item[2])
            bb = int(item[3])
            action_type = GraphAction(action_type_str, rxn=rxn, bb=bb)
            tuples_list.append((molecule, action_type))

        return tuples_list
    raise ValueError("String is not a valid GraphAction list representation")


def parse_graph_action_list_to_graphaction(s):
    # Define the regular expression pattern to match the list of tuples string
    # tuple_pattern = r"\('([^']*)', <GraphActionType\.([^,]+), >\)"
    tuple_pattern = r"\('([^']*)', <GraphActionType\.([^,]+), (\d+), (\d+)>"
    matches = re.findall(tuple_pattern, s)
    tuples_list = []
    if matches:
        tuples_list = []
        tuple_pattern = r"\('([^']*)', GraphActionType\.([^,]+), \)"

        for item in matches:
            molecule = item[0]
            action_type_str = item[1]
            action_type = GraphActionType[action_type_str]
            rxn = int(item[2])
            bb = int(item[3])
            tuples_list.append((molecule, GraphAction(action_type, rxn, bb)))

        return tuples_list
    raise ValueError("String is not a valid GraphAction list representation")


def read_db_file(db_path, deduplicate=False):
    db_path = os.path.abspath(db_path)
    conn = sqlite3.connect(db_path)
    query = "SELECT * from results"
    df = pd.read_sql_query(query, conn)
    df["smi"] = df["smi"].replace("", pd.NA)
    df = df.dropna(subset=["smi"])
    df["traj"] = df["traj"].apply(convert_to_strings_and_ints)
    df["traj_len"] = df["traj"].apply(lambda x: len(x))
    df["smi"] = df["smi"].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    if deduplicate:
        df = df[~df.duplicated(subset="smi", keep=False)]
        df = df.reset_index(drop=True)
    return df


def read_db_file_action_type(db_path, deduplicate=False):
    db_path = os.path.abspath(db_path)
    conn = sqlite3.connect(db_path)
    query = "SELECT * from results"
    df = pd.read_sql_query(query, conn)
    df["smi"] = df["smi"].replace("", pd.NA)
    df = df.dropna(subset=["smi"])
    df["traj"] = df["traj"].apply(parse_graph_action_list)
    df["traj_len"] = df["traj"].apply(lambda x: len(x) - 1)
    df["smi"] = df["smi"].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    if deduplicate:
        df = df[~df.duplicated(subset="smi", keep=False)]
        df = df.reset_index(drop=True)
    return df


def read_db_file_traj(db_path, deduplicate=False):
    db_path = os.path.abspath(db_path)
    conn = sqlite3.connect(db_path)
    query = "SELECT * from results"
    df = pd.read_sql_query(query, conn)
    df["smi"] = df["smi"].replace("", pd.NA)
    df = df.dropna(subset=["smi"])
    df["traj"] = df["traj"].apply(parse_graph_action_list_to_graphaction)
    df["traj_len"] = df["traj"].apply(lambda x: len(x) - 1)
    df["smi"] = df["smi"].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    if deduplicate:
        df = df[~df.duplicated(subset="smi", keep=False)]
        df = df.reset_index(drop=True)
    return df


def read_db_file_frag_model(db_path, drop_empty_smiles=True):
    db_path = os.path.abspath(db_path)
    conn = sqlite3.connect(db_path)
    query = "SELECT * from results"
    df = pd.read_sql_query(query, conn)
    if drop_empty_smiles:
        df["smi"] = df["smi"].replace("", pd.NA)
        df = df.dropna(subset=["smi"])
    df["smi"] = df["smi"].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    # df = df[~df.duplicated(subset="smi", keep=False)]
    df = df.reset_index(drop=True)
    return df


def trajectory_to_image(trajectory, r):
    smiles = []
    actions = []
    for t in trajectory:
        smiles.append(t[0])
        actions.append(t[1])
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    legends = []
    for i, a in enumerate(actions):
        if i == 0:
            legends.append("Building block. Next: T" + str(a))
        elif i == len(actions) - 1:
            legends.append(f"Reward= {r:.2f}")
        else:
            legends.append("Next: T" + str(a))
    img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(200, 200), legends=legends)
    return img, mols, legends


def plot_avg_reward_during_training(df_train, run_name, which_reward="r"):
    rewards = df_train[which_reward]
    # Calculate the average reward every 100 episodes
    episode_chunk_size = 64  # Number of episodes per chunk
    num_chunks = len(rewards) // episode_chunk_size
    average_rewards = [
        np.mean(rewards[i * episode_chunk_size : (i + 1) * episode_chunk_size]) for i in range(num_chunks)
    ]

    # Plot the average rewards
    plt.figure(figsize=(10, 6))
    plt.plot(average_rewards, marker="o", linestyle="-", color="b")
    plt.title(f"{run_name} Average Reward Every Episode")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.show()


def to_fingerprint(mol: Chem.Mol, finger_type: str = "morgan", path_len: int = 2, n_bits: int = 2000) -> np.ndarray:
    """
    Generate a fingerprint for a given RDKit molecule.

    Args:
        mol (Chem.Mol): The molecule for which to generate the fingerprint.
        finger_type (str): The type of fingerprint to generate. Options are 'morgan', 'rdkit', or 'maccs'.
        path_len (int): The path length parameter (relevant for 'morgan' and 'rdkit' fingerprints).
        n_bits (int): The size of the fingerprint bit vector (relevant for 'morgan' and 'rdkit' fingerprints).

    Returns:
        np.ndarray: The generated fingerprint as a NumPy array.

    Raises:
        ValueError: If an unsupported fingerprint type is provided.
    """

    if finger_type == "morgan":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, path_len, nBits=n_bits)
    elif finger_type == "rdkit":
        fp = Chem.RDKFingerprint(mol, maxPath=path_len, fpSize=n_bits)
    elif finger_type == "maccs":
        fp = Chem.MACCSkeys.GenMACCSKeys(mol)
    else:
        raise ValueError(f"Unsupported fingerprint type: {finger_type}. Supported types are morgan, rdkit, or maccs.")

    # Convert the RDKit ExplicitBitVect to a NumPy array
    return np.array(fp, dtype=int)


def to_bitvector(fp: np.ndarray) -> ExplicitBitVect:
    """
    Convert a fingerprint to an RDKit ExplicitBitVect.

    Args:
    fp (np.ndarray): Fingerprint as a NumPy array.

    Returns:
    rdkit.DataStructs.cDataStructs.ExplicitBitVect: Fingerprint as an RDKit ExplicitBitVect.
    """
    # Initialize ExplicitBitVect with all bits set to 0
    num_bits = fp.shape[0]
    ebv = DataStructs.ExplicitBitVect(num_bits)

    # Set bits that are set to 1 in the fingerprint
    for i in range(num_bits):
        if fp[i]:
            ebv.SetBit(i)

    return ebv


def standardize_molecules(molecules: List[Chem.Mol], remove_conformers: bool = True) -> List[Chem.Mol]:
    """
    Standardize a list of RDKit molecules by sanitizing them, removing hydrogens,
    and removing all conformers.

    Args:
    - molecules (List[Chem.Mol]): List of RDKit Molecule objects.
    - remove_conformers (bool): Whether to remove all conformers from the molecules. Defaults to True.

    Returns:
    - List[Chem.Mol]: List of standardized RDKit Molecule objects.
    """
    standardized_molecules = [dm.sanitize_mol(mol) for mol in molecules]
    standardized_molecules = [dm.remove_hs(mol) for mol in standardized_molecules]
    if remove_conformers:
        for mol in standardized_molecules:
            mol.RemoveAllConformers()
    return standardized_molecules


def calculate_molecular_diversity(molecules, fingerprint_type="morgan"):
    """
    Calculate molecular diversity of a list of RDKit molecule objects.

    Parameters:
    molecules (list of rdkit.Chem.Mol objects): List of RDKit molecules.
    fingerprint_type (str): Type of fingerprint to use. Options are 'morgan', 'rdkit', or 'maccs'. Defaults to 'morgan'.

    Returns:
    float: Average pairwise Tanimoto dissimilarity of the molecules.
    """
    # Generate fingerprints for each molecule
    fps = [to_fingerprint(mol, finger_type=fingerprint_type) for mol in molecules]

    # Convert fingerprints to numpy array
    np_fps = np.array([np.array(fp) for fp in fps])

    # Calculate pairwise Tanimoto dissimilarities
    tanimoto_dissimilarity = pdist(np_fps, metric="jaccard")

    # Return average dissimilarity
    return np.mean(tanimoto_dissimilarity), tanimoto_dissimilarity


def calculate_tanimoto_similarity(molecules):
    fps = [to_bitvector(to_fingerprint(mol)) for mol in molecules]
    similarity_scores = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            similarity = DataStructs.FingerprintSimilarity(fps[i], fps[j])
            similarity_scores.append(similarity)

    average_similarity = np.mean(similarity_scores)
    return average_similarity, similarity_scores


def find_centroid_molecule(cluster_id: int, cluster_ids: List[int], fingerprints: np.array) -> int:
    """
    Identify the molecule in a specified cluster that is closest to the centroid of the cluster in fingerprint space.

    Args:
    cluster_id (int): The ID of the cluster to analyze.
    cluster_ids (List[int]): List of cluster IDs corresponding to each fingerprint.
    fingerprints (np.array): Fingerprints of the molecules as a NumPy array.

    Returns:
    int: Index of the molecule that is closest to the centroid of the specified cluster.
    """
    # Make array if list
    if isinstance(fingerprints, list):
        fingerprints = np.array(fingerprints)

    # Initialize a fingerprint for the centroid with all bits set to 0
    num_bits = fingerprints.shape[1]
    centroid_fp = np.zeros(num_bits, dtype=int)

    # Determine which fingerprints belong to the specified cluster
    cluster_indices = [idx for idx, cid in enumerate(cluster_ids) if cid == cluster_id]

    # Sum the bits for each position for fingerprints in the cluster
    for idx in cluster_indices:
        centroid_fp |= fingerprints[idx]

    # Average the bits (set bit in centroid if it's set in more than half of the fingerprints in the cluster)
    centroid_fp = np.where(centroid_fp > len(cluster_indices) / 2, 1, 0)

    # Initialize minimum distance and index of centroid molecule
    min_distance = float("inf")
    centroid_molecule_idx = None

    # Convert centroid_fp to ExplicitBitVect for TanimotoSimilarity calculation
    centroid_ebv = DataStructs.ExplicitBitVect(num_bits)
    for i in range(num_bits):
        if centroid_fp[i]:
            centroid_ebv.SetBit(i)

    # Iterate through each molecule in the cluster
    for idx in cluster_indices:
        # Convert molecule fingerprint to ExplicitBitVect
        molecule_ebv = DataStructs.ExplicitBitVect(num_bits)
        for i in range(num_bits):
            if fingerprints[idx, i]:
                molecule_ebv.SetBit(i)
        # Calculate Tanimoto similarity between molecule and centroid
        distance = 1 - DataStructs.TanimotoSimilarity(centroid_ebv, molecule_ebv)
        # Update minimum distance and index if current molecule is closer to the centroid
        if distance < min_distance:
            min_distance = distance
            centroid_molecule_idx = idx

    return centroid_molecule_idx


def find_centroid_molecules(cluster_ids: List[int], fingerprints: np.array) -> List[int]:
    """
    Identify the molecule in each cluster that is closest to the centroid of the cluster in fingerprint space.

    Args:
        cluster_ids (List[int]): List of cluster IDs corresponding to each fingerprint.
        fingerprints (np.array): Fingerprints of the molecules as a NumPy array.

    Returns:
        List[int]: List of indices of the molecules that are closest to the centroid of each cluster.
    """

    # Find the unique cluster IDs
    unique_cluster_ids = np.unique(cluster_ids)

    # Initialize list of centroid molecule indices
    centroid_molecule_indices = []

    # Iterate through each cluster
    for cluster_id in unique_cluster_ids:
        # Find the index of the molecule in the cluster that is closest to the centroid
        centroid_molecule_idx = find_centroid_molecule(cluster_id, cluster_ids, fingerprints)
        # Append the index to the list of centroid molecule indices
        centroid_molecule_indices.append(centroid_molecule_idx)

    return centroid_molecule_indices


def butina_clustering(fps, cutoff=0.5) -> List[List[int]]:
    """
    Perform Butina clustering on a list of fingerprints.

    Args:
        fps (List[ExplicitBitVect]): List of fingerprints as RDKit ExplicitBitVect objects.
        cutoff (float): The Tanimoto similarity cutoff to use for clustering.

    Returns:
        List[List[int]]: List of clusters, where each cluster is a list of
        indices of the molecules in that cluster.
    """

    # Convert to explicit vectors if necessary
    if isinstance(fps[0], np.ndarray):
        fps = [to_bitvector(x) for x in fps]

    # first generate the distance matrix:
    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])

    # now cluster the data:
    clusters = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)

    # Convert clusters to a list of cluster IDs
    cluster_ids = [-1] * nfps  # Initialize with -1 (or any placeholder for unassigned)
    for cluster_id, cluster_indices in enumerate(clusters):
        for idx in cluster_indices:
            cluster_ids[idx] = cluster_id  # Assign the cluster ID

    return cluster_ids


def get_centroid_for_top_k_clusters(cluster_ids, molecules, fingerprints, k=5):
    """
    Identify the molecule in each of the top k clusters that is closest to the centroid of the cluster in fingerprint space.

    Returns the molecules corresponding to the centroid molecule indices.

    Args:
        cluster_ids (List[int]): List of cluster IDs corresponding to each fingerprint.
        molecules (List[Chem.Mol]): List of RDKit molecules.
        fingerprints (np.array): Fingerprints of the molecules as a NumPy array.
        k (int): The number of clusters to analyze.

    Returns:
        List[Chem.Mol]: List of molecules corresponding to the centroid molecules.
    """

    # Find the unique cluster IDs
    series = pd.Series(cluster_ids)
    unique_cluster_ids = series.value_counts().index

    # Initialize list of centroid molecule indices
    centroid_molecule_indices = []

    # Iterate through each cluster
    for cluster_id in unique_cluster_ids[:k]:
        # Find the index of the molecule in the cluster that is closest to the centroid
        centroid_molecule_idx = find_centroid_molecule(cluster_id, cluster_ids, fingerprints)
        # Append the index to the list of centroid molecule indices
        centroid_molecule_indices.append(centroid_molecule_idx)

    # Get the molecules corresponding to the centroid molecule indices
    centroid_molecules = [molecules[idx] for idx in centroid_molecule_indices]

    return centroid_molecules


def to_canonical_smiles(mol: Chem.Mol) -> str:
    """
    Convert an RDKit molecule to a canonical SMILES string.

    Args:
        mol (Chem.Mol): The molecule to convert.

    Returns:
        str: The canonical SMILES string.
    """
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


def mols_to_canonical_smiles(mols: List[Chem.Mol]) -> List[str]:
    """
    Convert a list of RDKit molecules to a list of canonical SMILES strings.

    Args:
        mols (List[Chem.Mol]): The molecules to convert.

    Returns:
        List[str]: The canonical SMILES strings.
    """
    return [to_canonical_smiles(mol) for mol in mols]

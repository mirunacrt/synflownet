import ast
import os
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


def convert_to_strings_and_ints(input_str):
    tuples_list = ast.literal_eval(input_str)
    results = []
    for item in tuples_list:
        result_string = item[0]
        result_int = item[1]
        results.append((result_string, result_int))
    return results


def read_db_file(db_path):
    db_path = os.path.abspath(db_path)
    conn = sqlite3.connect(db_path)
    query = "SELECT * from results"
    df = pd.read_sql_query(query, conn)
    df["traj"] = df["traj"].apply(convert_to_strings_and_ints)
    df["traj_len"] = df["traj"].apply(lambda x: len(x))
    df["smi"] = df["smi"].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    # df = df[~df.duplicated(subset="traj", keep=False)]
    df = df.reset_index(drop=True)
    return df


def read_db_file_frag_model(db_path):
    db_path = os.path.abspath(db_path)
    conn = sqlite3.connect(db_path)
    query = "SELECT * from results"
    df = pd.read_sql_query(query, conn)
    df["smi"] = df["smi"].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x)))
    # df = df[~df.duplicated(subset="smi", keep=False)]
    df = df.reset_index(drop=True)
    return df


def plot_avg_reward_during_training(df_train, run_name, which_reward="r"):
    rewards = df_train[which_reward]
    # Calculate the average reward every 100 episodes
    episode_chunk_size = 100
    num_chunks = len(rewards) // episode_chunk_size
    average_rewards = [
        np.mean(rewards[i * episode_chunk_size : (i + 1) * episode_chunk_size]) for i in range(num_chunks)
    ]

    # Plot the average rewards
    plt.figure(figsize=(10, 6))
    plt.plot(average_rewards, marker="o", linestyle="-", color="b")
    plt.title(f"{run_name} Average Reward Every 100 Episodes")
    plt.xlabel("Chunk Number (each represents 100 trajectories)")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.show()


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
    return img


script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
print(script_dir, repo_root)
DB_PATH = os.path.join(repo_root, "tasks/logs/debug_run_seh_reactions/final/", "generated_mols_0.db")


def main(db_path: str = DB_PATH, number_of_examples: int = 6):
    df = read_db_file(db_path)
    max_traj_len = df["traj_len"].max()
    indices_of_max_traj_len = df.index[df["traj_len"] == max_traj_len].tolist()

    examples = 0
    paths = [os.path.join(script_dir, f"example_trajectory_{i}.png") for i in range(0, number_of_examples)]
    for i, n in enumerate(indices_of_max_traj_len):
        if i == number_of_examples:
            break
        img = trajectory_to_image(df["traj"][n], round(df["r"][n], 2))
        img.save(paths[i])
        examples += 1

    while examples < number_of_examples:
        max_traj_len -= 1
        indices = df.index[df["traj_len"] == max_traj_len].tolist()
        for i, n in enumerate(indices):
            img = trajectory_to_image(df["traj"][n], round(df["r"][n], 2))
            img.save(paths[examples])
            examples += 1
            if examples == number_of_examples:
                break

    return paths


if __name__ == "__main__":
    main()

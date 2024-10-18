import os
import sys
import time

import torch
import wandb

from gflownet.config import Config, init_empty
from gflownet.tasks.reactions_task import ReactionTrainer

TIME = time.strftime("%m-%d-%H-%M")
ENTITY = "valencelabs"
PROJECT = "gflownet"
SWEEP_NAME = f"{TIME}-sehreactions"
STORAGE_DIR = f"~/storage/wandb_sweeps/{SWEEP_NAME}"


# Define the search space of the sweep
sweep_config = {
    "name": SWEEP_NAME,
    "program": "init_wandb_sweep.py",
    "controller": {
        "type": "cloud",
    },
    "method": "grid",
    "parameters": {
        "config.seed": {"values": [0, 1, 2]},
    },
}


def wandb_config_merger():
    config = init_empty(Config())
    wandb_config = wandb.config

    # Set desired config values
    config.log_dir = f"{STORAGE_DIR}/{wandb.run.name}-id-{wandb.run.id}"
    config.print_every = 100
    config.validate_every = 500
    config.num_final_gen_steps = 100
    config.num_training_steps = 5000
    config.pickle_mp_messages = False
    config.overwrite_existing_exp = False
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.num_workers = 8
    config.algo.sampling_tau = 0.99
    config.algo.train_random_action_prob = 0.02
    config.algo.tb.Z_learning_rate = 1e-3
    config.algo.tb.Z_lr_decay = 2_000
    config.algo.tb.do_parameterize_p_b = True
    config.algo.tb.do_sample_p_b = False
    config.algo.max_len = 3
    config.algo.tb.backward_policy = "MaxLikelihood"
    config.algo.strict_bck_masking = False

    config.model.graph_transformer.fingerprint_type = "morgan_1024"
    config.model.graph_transformer.continuous_action_embs = True

    config.cond.temperature.sample_dist = "constant"
    config.cond.temperature.dist_params = [32.0]
    config.cond.weighted_prefs.preference_type = "dirichlet"
    config.cond.focus_region.focus_type = None
    config.replay.use = False

    config.reward = "seh_reaction"

    # Merge the wandb sweep config with the nested config from gflownet
    config.seed = wandb_config["config.seed"]

    return config


if __name__ == "__main__":
    # if there no arguments, initialize the sweep, otherwise this is a wandb agent
    if len(sys.argv) == 1:
        if os.path.exists(STORAGE_DIR):
            raise ValueError(f"Sweep storage directory {STORAGE_DIR} already exists.")

        wandb.sweep(sweep_config, entity=ENTITY, project=PROJECT)

    else:
        wandb.init(entity=ENTITY, project=PROJECT)
        config = wandb_config_merger()
        trial = ReactionTrainer(config)
        trial.run()

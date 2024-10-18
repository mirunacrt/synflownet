import argparse
import datetime
import os
import pickle
import shutil
import socket
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import rdkit.Chem as Chem
import torch
import torch.nn as nn
import torch_geometric.data as gd
import wandb
import yaml
from rdkit.Chem import QED
from rdkit.Chem.rdchem import Mol as RDMol
from tdc import Oracle
from torch import Tensor
from torch_scatter import scatter_add

from gflownet import GFNTask, LogScalar, ObjectProperties
from gflownet.algo.reaction_sampling import SynthesisSampler
from gflownet.config import Config, init_empty, load_yaml, override_config
from gflownet.envs.synthesis_building_env import ReactionTemplateEnv, ReactionTemplateEnvContext
from gflownet.models import bengio2021flow
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.utils import metrics, sascore
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.utils.gpu_vina import QuickVina2GPU
from gflownet.utils.misc import get_worker_device
from gflownet.utils.transforms import to_logreward

# Write any local overrides required here
LOCAL_OVERRIDES = "local_overrides.yaml"
from gflownet.envs.graph_building_env import Graph


class ReactionTask(GFNTask):
    """Sets up a task where the reward is computed using a proxy for the binding energy of a molecule to
    Soluble Epoxide Hydrolases.

    The proxy is pretrained, and obtained from the original GFlowNet paper, see `gflownet.models.bengio2021flow`.

    This setup essentially reproduces the results of the Trajectory Balance paper when using the TB
    objective, or of the original paper when using Flow Matching.
    """

    def __init__(
        self,
        cfg: Config,
        wrap_model: Callable[[nn.Module], nn.Module] = None,
    ):
        self._wrap_model = wrap_model
        self.cfg = cfg
        self.reward = cfg.reward
        self.models = self._load_task_models()
        self.temperature_conditional = TemperatureConditional(cfg)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

    def _load_task_models(self):
        model = bengio2021flow.load_original_model()
        model.to(get_worker_device())
        model = self._wrap_model(model)
        return {"seh": model}

    def sample_conditional_information(self, n: int, train_it: int, final: bool = False) -> Dict[str, Tensor]:
        if final:
            cfg = self.cfg
            cfg.cond.temperature.sample_dist = "constant"
            cfg.cond.temperature.dist_params = [64.0]
            self.temperature_conditional = TemperatureConditional(cfg)
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: ObjectProperties) -> LogScalar:
        return LogScalar(self.temperature_conditional.transform(cond_info, to_logreward(flat_reward)))

    def seh_proxy_reward_from_graph(self, graphs: List[gd.Data], mols: List[Chem.Mol], traj_lens: Tensor) -> Tensor:
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        batch.to(self.models["seh"].device if hasattr(self.models["seh"], "device") else get_worker_device())
        preds = self.models["seh"](batch).reshape((-1,)).data.cpu() / 8
        if self.cfg.task.reactions_task.reward is None:
            # Using only affinity proxy
            pass
        elif self.cfg.task.reactions_task.reward == "efficiency":
            mol_sizes = scatter_add(torch.ones_like(batch.batch), batch.batch, dim=0).cpu()
            avg_sizes = traj_lens * 13  # 13 is the average size of the BBs from large set
            norm = (mol_sizes / avg_sizes) + 1e-5
            preds = preds / norm
        elif self.cfg.task.reactions_task.reward.startswith("size_penalty"):
            _, _, max_size, penalty = self.cfg.task.reactions_task.reward.split("_")
            mol_sizes = scatter_add(torch.ones_like(batch.batch), batch.batch, dim=0).cpu()
            penalties = (mol_sizes > int(max_size)) * float(penalty)
            preds = preds - penalties
        else:
            raise NotImplementedError

        preds[preds.isnan()] = 0
        return preds.clip(1e-4, 100).reshape((-1,))

    def safe(self, f, x, default):
        try:
            return f(x)
        except Exception:
            return default

    def vina_rewards_from_graph(self, graphs: List[gd.Data], mols: List[Chem.Mol]) -> Tensor:

        print("Calculating Vina rewards...")
        vina = QuickVina2GPU(vina_path=self.cfg.vina.vina_path, target=self.cfg.vina.target)
        # Convert smiles to mols
        smiles = [Chem.MolToSmiles(mol) for mol in mols]
        _, _, vina_rewards = vina.calculate_rewards(smiles)

        return torch.tensor(vina_rewards).float().clip(1e-4, 100)

    def qed_rewards_from_mols(self, mols: List[RDMol]) -> Tensor:
        return torch.tensor([QED.qed(m) for m in mols])

    def gsk_rewards_from_graph(self, graphs: List[gd.Data], mols: List[Chem.Mol]) -> Tensor:
        gsk = Oracle("GSK3B")
        smiles = [Chem.MolToSmiles(mol) for mol in mols]
        gsk_rewards = [gsk(s) for s in smiles]

        return torch.tensor(gsk_rewards).float().clip(1e-4, 100)

    def drd2_rewards_from_graph(self, graphs: List[gd.Data], mols: List[Chem.Mol]) -> Tensor:
        drd2 = Oracle("DRD2")
        smiles = [Chem.MolToSmiles(mol) for mol in mols]
        drd2_rewards = [drd2(s) for s in smiles]

        return torch.tensor(drd2_rewards).float().clip(1e-4, 100)

    def mol2sas(self, mols: list[RDMol], default=10):
        sas = torch.tensor([self.safe(sascore.calculateScore, i, default) for i in mols])
        # sas = (10 - sas) / 9  # Turn into a [0-1] reward
        sas = ((10 - sas) / (10 - 3.5)).clamp(0, 1)
        return torch.tensor(sas).float().clip(1e-4, 100)

    def compute_reward_from_graph(self, graphs: List[gd.Data], mols: List[Chem.Mol], traj_lens: Tensor) -> Tensor:
        if self.reward == "seh_reaction":
            return self.seh_proxy_reward_from_graph(graphs, mols, traj_lens)
        elif self.reward == "vina":
            return self.vina_rewards_from_graph(graphs, mols)
        elif self.reward == "qed":
            return self.qed_rewards_from_mols(mols)
        elif self.reward == "gsk":
            return self.gsk_rewards_from_graph(graphs, mols)
        elif self.reward == "drd2":
            return self.drd2_rewards_from_graph(graphs, mols)
        elif self.reward == "reward_one":
            return torch.ones(len([g for g in graphs if g is not None]))
        elif self.reward == "seh_qed":
            seh_scores = self.seh_proxy_reward_from_graph(graphs, mols, traj_lens)
            qed_scores = self.qed_rewards_from_mols(mols)
            qed_scores = (qed_scores / 0.7).clamp(0, 1)
            return seh_scores * qed_scores
        elif self.reward == "seh_sa":
            seh_scores = self.seh_proxy_reward_from_graph(graphs, mols, traj_lens)
            sas = self.mol2sas(mols)
            return seh_scores * sas
        elif self.reward == "seh_mol_wt":
            seh_scores = self.seh_proxy_reward_from_graph(graphs, mols, traj_lens)
            for i, mol in enumerate(mols):
                # Penalize molecules with more than reference number of atoms
                if mol.GetNumHeavyAtoms() > 37:
                    seh_scores[i] += -0.4
            return seh_scores
        else:
            raise ValueError(f"Unknown task {self.task}")

    def compute_obj_properties(self, mols: List[RDMol], traj_lens: Tensor, **kwargs) -> Tuple[ObjectProperties, Tensor]:
        graphs = []
        for i in mols:
            try:
                g = bengio2021flow.mol2graph(i)
            except IndexError:
                # NOTE: affinity model does not have all atom types found in BBs
                # so for now we mark such molecules as invalid
                g = None
            graphs.append(g)

        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return ObjectProperties(torch.zeros((0, 1))), is_valid

        preds = self.compute_reward_from_graph(graphs=graphs, mols=mols, traj_lens=traj_lens).reshape((-1, 1))
        assert len(preds) == is_valid.sum(), f"len(preds)={len(preds)}, is_valid.sum()={is_valid.sum()}"

        return ObjectProperties(preds), is_valid


class ReactionTrainer(StandardOnlineTrainer):
    task: ReactionTask

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.pickle_mp_messages = False
        cfg.num_workers = 8
        cfg.opt.learning_rate = 1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20_000
        cfg.opt.clip_grad_type = "norm"
        cfg.opt.clip_grad_param = 10
        cfg.algo.num_from_policy = 64
        cfg.model.num_emb = 128
        cfg.model.num_layers = 4

        cfg.algo.method = "TB"
        cfg.algo.max_nodes = 9
        cfg.algo.sampling_tau = 0.9
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = 0.0
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.valid_num_from_policy = 64
        cfg.num_validation_gen_steps = 10
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-3
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_parameterize_p_b = False
        cfg.algo.tb.do_sample_p_b = False

        cfg.replay.use = False
        cfg.replay.capacity = 10_000
        cfg.replay.warmup = 64
        cfg.replay.num_from_replay = 0
        cfg.replay.num_new_samples = 64

    def setup_task(self):
        self.task = ReactionTask(
            cfg=self.cfg,
            wrap_model=self._wrap_for_mp,
        )

    def setup_env_context(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)

        # Load building blocks
        building_blocks_path = os.path.join(
            repo_root, "data/building_blocks", self.cfg.task.reactions_task.building_blocks_filename
        )
        with open(building_blocks_path, "r") as file:
            building_blocks = file.read().splitlines()

        # Load templates
        templates_path = os.path.join(repo_root, "data/templates", self.cfg.task.reactions_task.templates_filename)
        with open(templates_path, "r") as file:
            reaction_templates = file.read().splitlines()

        precomputed_bb_masks_path = os.path.join(
            repo_root, "data/building_blocks", self.cfg.task.reactions_task.precomputed_bb_masks_filename
        )
        with open(precomputed_bb_masks_path, "rb") as f:
            precomputed_bb_masks = pickle.load(f)

        self.ctx = ReactionTemplateEnvContext(
            num_cond_dim=self.task.num_cond_dim,
            building_blocks=building_blocks,
            reaction_templates=reaction_templates,
            precomputed_bb_masks=precomputed_bb_masks,
            fp_type=self.cfg.model.graph_transformer.fingerprint_type,
            fp_path=self.cfg.model.graph_transformer.fingerprint_path,
            strict_bck_masking=self.cfg.algo.strict_bck_masking,
        )
        self.env = ReactionTemplateEnv(
            reaction_templates=reaction_templates,
            building_blocks=building_blocks,
            precomputed_bb_masks=precomputed_bb_masks,
            fp_type=self.cfg.model.graph_transformer.fingerprint_type,
            fp_path=self.cfg.model.graph_transformer.fingerprint_path,
        )

    def setup_sampler(self):
        self.sampler = SynthesisSampler(
            cfg=self.cfg,
            ctx=self.ctx,
            env=self.env,
            max_len=self.cfg.algo.max_len,
            correct_idempotent=self.cfg.algo.tb.do_correct_idempotent,
            pad_with_terminal_state=self.cfg.algo.tb.do_parameterize_p_b,
        )

    def build_callbacks(self):

        from rdkit.Chem.Scaffolds import MurckoScaffold

        graph_to_obj = self.ctx.graph_to_obj
        reward_fn = lambda rdmols, traj_lens: self.task.compute_obj_properties(rdmols, traj_lens)[0]

        class UniqueMurckoScaffoldsCallback:
            def __init__(self, reward_thresh):
                self._reward_thresh = reward_thresh

            def on_validation_end(self, step_outputs):

                mols = []
                rewards = []
                for out in step_outputs:
                    batch = out["batch"]
                    final_graph_idx = torch.cumsum(batch.traj_lens, 0) - 1
                    final_graphs = [batch.nx_graphs[i] for i in final_graph_idx]
                    batch_mols = [graph_to_obj(g) for g in final_graphs]
                    mols.extend(batch_mols)
                    rewards.append(reward_fn(batch_mols, batch.traj_lens))

                murcko_scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]
                rewards = torch.cat(rewards)
                assert len(murcko_scaffolds) == len(rewards)

                scaffolds_above_thresh = [smi for smi, r in zip(murcko_scaffolds, rewards) if r > self._reward_thresh]
                unique_scaffolds = set(scaffolds_above_thresh)

                return {f"unique_murcko_r_gt_{self._reward_thresh}": len(unique_scaffolds)}

        return {"murcko_scaffolds": UniqueMurckoScaffoldsCallback(reward_thresh=0.5)}


def main(wandb_run_name, backoff=False):
    """Example of how this trainer can be run"""

    name = None
    checkpoint_path = None
    if backoff:
        assert wandb_run_name is not None
        logroot = "./logs"
        logdirs = [d for d in os.listdir(logroot) if d.startswith(wandb_run_name)]
        if len(logdirs) > 0:
            name = sorted(logdirs)[0]
            checkpoint_path = os.path.join(logroot, name, "model_state.pt")
            wandb.init(project="synflownet", name=name, id=name, resume="must")
            print(f"Found checkpoint at {checkpoint_path}")

    if name is None and checkpoint_path is None:
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if wandb_run_name is not None:
            name = f"{wandb_run_name}_{now}"
            wandb.init(project="synflownet", name=name, id=name)
        else:
            name = f"debug_{now}"
        print(f"No checkpoint found, starting a new run {name}")

    # trainer is loaded from checkpoint using StandardOnlineTrainer.load_from_checkpoint
    if checkpoint_path is not None:
        trial = ReactionTrainer.load_from_checkpoint(checkpoint_path)
    else:
        config = init_empty(Config())
        config.reward = "seh_reaction"  # vina, seh_reaction, gsk, drd2, seh_qed
        config.print_every = 1
        config.log_dir = f"./logs/debug_run_reactions_task_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
        config.overwrite_existing_exp = True
        config.num_training_steps = 5000
        config.validate_every = 500
        config.num_final_gen_steps = 100
        config.num_workers = 0
        config.opt.learning_rate = 1e-4
        config.opt.lr_decay = 2_000
        config.algo.sampling_tau = 0.99
        config.cond.temperature.sample_dist = "constant"
        config.cond.temperature.dist_params = [32.0]
        config.model.graph_transformer.continuous_action_embs = True
        config.model.graph_transformer.fingerprint_type = "morgan_1024"

        # For Vina experiments
        config.vina.target = "kras"

        # Load local overrides
        if os.path.exists(LOCAL_OVERRIDES):
            overrides = load_yaml(LOCAL_OVERRIDES)
            config = override_config(config, overrides)

        # Activate WandB here if not running experiment through a sweep
        if hasattr(config, "wandb"):
            wandb.init(project=config.wandb.project, entity=config.wandb.entity, config=config)
        config.algo.max_len = 3
        config.algo.tb.backward_policy = "MaxLikelihood"
        config.algo.tb.do_parameterize_p_b = True
        config.algo.tb.do_sample_p_b = False
        config.replay.use = False
        config.algo.synthesis_cost_as_bck_reward = False
        config.algo.tb.reinforce_loss_multiplier = 1.0
        config.algo.tb.bck_entropy_loss_multiplier = 1.0
        config.algo.num_from_policy = 64
        config.algo.num_from_buffer_for_pb = 64

        config.algo.strict_forward_policy = False  # If True, a fwd action is allowed only if in reverse it produces the exact same reaction (identical reactants and products)
        config.algo.strict_bck_masking = (
            False  # If True, bimolecular bck actions are masked if they don't produce at least one bb
        )
        config.algo.tb.do_correct_idempotent = False

        trial = ReactionTrainer(config)
    trial.run()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wandb_run_name", type=str, required=False, default=None)
    p.add_argument("--backoff", action="store_true", required=False, default=False)
    args = p.parse_args()
    main(args.wandb_run_name, args.backoff)

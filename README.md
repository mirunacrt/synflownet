

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2405.01155)
[![Python versions](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

![GFlowNet](docs/synflownet_logo.png)

# SynFlowNet - Design of Diverse and Novel Molecules with Synthesis Constraints

Official implementation of SynFlowNet, a GFlowNet model with a synthesis action space. The paper is available on [arxiv](https://arxiv.org/abs/2405.01155).

**Primer**

SynFlowNet is a GFlowNet model that generates molecules from chemical reactions and available building blocks. This repo contains instructions for how to train SynFlowNet and sample synthesisable molecules with probability proportional to a reward specified by the user. The code builds upon the [recursionpharma/gflownet](https://github.com/recursionpharma/gflownet) codebase, available under the [MIT](https://github.com/recursionpharma/gflownet/blob/trunk/LICENSE) license. For a primer and repo overview visit [recursionpharma/gflownet](https://github.com/recursionpharma/gflownet).

![GFlowNet](docs/concept.png)

## Installation

### PIP

This package is installable as a PIP package, but since it depends on some torch-geometric package wheels, the `--find-links` arguments must be specified as well:

```bash
conda create --name sfn python=3.10.14
conda activate sfn
pip install -e . --find-links https://data.pyg.org/whl/torch-2.1.2+cu121.html
```
Or for CPU use:

```bash
pip install -e . --find-links https://data.pyg.org/whl/torch-2.1.2+cpu.html
```

## Getting started

### Data

The training relies on two data sources: reaction templates and building blocks. Filenames are specified in the `ReactionTaskConfig`. The model uses pre-computed masks to ensure compatibility between the building blocks and the reaction templates. Instructions for preprocessing building blocks and for computing masks can be found in [synflownet/data/building_blocks](https://github.com/recursionpharma/synflownet/tree/miruna-cleanup/synflownet/data/building_blocks).

### Reward

SynFlowNet uses a reward to guide the molecule generation process. We have implemented a few reward functions in the `ReactionTask` class. These include the SeH binding proxy, QED, oracles from [PyTDC](https://pypi.org/project/PyTDC/) and Vina docking (see below). Other reward functions can be imported in the `synflownet/tasks/reactions_task.py` file.

### Training

The model can be trained by running `synflownet/tasks/reactions_task.py` using different reward functions. You may want to change the default configuration in `main()`.

#### [Optional] If using GPU-accelerated Vina

For easy adoption to other targets, a GPU-accelerated version of Vina docking can be used to calculate rewards as binding affinities to targets of interest. Follow the instructions at [this repo](https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1) to compile an excuteable for `QuickVina2-GPU-2-1`. One done, place the excuteable in `bin/`.

## Implementation notes

Below is a breakdown of the different components of SynFlowNet. 

### Environment, Context, Task, Trainers

We separate experiment concerns in four categories:

- `ReactionTemplateEnv` is the definition of the reaction MDP and it implements stepping forward and backward in the environment.
- `ReactionTemplateEnvContext` provides an interface between the agent and the environment, it
  - maps graphs to other molecule representations and to torch_geometric `Data` instances 
  - maps GraphActions to action indices
  - creates masks for actions
  - communicates with the model what inputs it should expect
- The `ReactionTask` class is responsible for computing rewards, and for sampling conditional information
- The `ReactionTrainer` class is responsible for instanciating everything, and running the training loop

### Policies and action categoricals

The `GraphTransformerSynGFN` class is used to parameterize the policies and outputs a specific categorical distribution type for the actions defined in `ReactionTemplateEnvContext`. If `config.model.graph_transformer.continuous_action_embs` is set to `True`, then the probability of sampling building blocks is computed from the normalized dot product of the molecule representation and the embedding vector of the state. The `ActionCategorical` class contains the logic to sample from the hierarchical distribution of actions. 

### Data sources

The data used for training the GFlowNet can come from multiple sources:
- Generating new trajectories on-policy from _s_0_
- Generating new trajectories on-policy backwards from samples stored in a replay buffer (for training the backwards policy with REINFORCE)
- Sampling trajectories from a fixed, offline dataset

# Citation

If you use this code in your research, please cite the following paper:

```
@article{cretu2024synflownet,
      title={SynFlowNet: Design of Diverse and Novel Molecules with Synthesis Constraints},
      author={Miruna Cretu, Charles Harris, Ilia Igashov, Arne Schneuing, Marwin Segler, Bruno Correia, Julien Roy, Emmanuel Bengio and Pietro Liò},
      journal={arXiv preprint arXiv},
      year={2024}
}
```

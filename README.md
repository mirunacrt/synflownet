

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2405.01155)
[![Python versions](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

![GFlowNet](docs/synflownet_logo.png)

# SynFlowNet - Design of Diverse and Novel Molecules with Synthesis Constraints

Official implementation of SynFlowNet, a GFlowNet model with a synthesis action space.

**Primer**

SynFlowNet is a GFlowNet model that generates molecules from chemical reactions and available building blocks. SynFlowNet is trained to sample molecules with probabilities proportional to their rewards. This repo contains instructions for how to train SynFlowNet and sample synthesisable molecules. The code builds upon the codebase provided by [recursionpharma/gflownet](https://github.com/recursionpharma/gflownet), available under the [MIT](https://github.com/recursionpharma/gflownet/blob/trunk/LICENSE) license. For a primer and repo overview visit [recursionpharma/gflownet](https://github.com/recursionpharma/gflownet).

![GFlowNet](docs/concept.png)

## Installation

### PIP

This package is installable as a PIP package, but since it depends on some torch-geometric package wheels, the `--find-links` arguments must be specified as well:

```bash
pip install -e . --find-links https://data.pyg.org/whl/torch-2.1.2+cu121.html
```
Or for CPU use:

```bash
pip install -e . --find-links https://data.pyg.org/whl/torch-2.1.2+cpu.html
```

## Getting started

### Data

The training relies on two data sources: modified _Hartenfeller-Button_ reaction templates and _Enamine_ building blocks. The building blocks are not freely available and can be obtained upon request from [enamine.net/building-blocks/building-blocks-catalog](https://enamine.net/building-blocks/building-blocks-catalog). Instructions can be found in `src/gflownet/data/building_blocks/`.

### Training

The model can be trained by running `src/gflownet/tasks/reactions_task.py` using different reward functions implemented in the same file. You may want to change the default configuration in `main()`.

#### [Optional] If using GPU-accelerated Vina

For easy adoption to other targets, a GPU-accelerated version of Vina docking can be used to calculate rewards as binding affinities to targets of interest. Follow the instructions at [this repo](https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1) to compile an excuteable for `QuickVina2-GPU-2-1`. One done, place the excuteable in `bin/`.

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

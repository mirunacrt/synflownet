# Implementation notes

(Updated from [recursionpharma/gflownet](https://github.com/recursionpharma/gflownet/blob/trunk/docs/implementation_notes.md))

This repo is centered around training GFlowNets that produce molecules from sequentially applying chemical reactions to reactants (building blocks). The building blocks were made available by Enamine upon request and the reaction templates are modified _Hartenfeller-Button_ reaction templates. 

## Environment, Context, Task, Trainers

- We implemented a new Environment, which defines an MDP which starts from an empty graph, followed by an Enamine building block. Stepping forward in the environment consists in running a reaction using RDKit.
- The Context provides an interface between the agent and the environment, it 
    - maps graphs to torch_geometric `Data` 
  instances
    - maps GraphActions to action indices
    - produces action masks
    - communicates to the model what inputs it should expect
- The Task class is responsible for computing the reward of a state, and for sampling conditioning information 
- The Trainer class is responsible for instanciating everything, and running the training & testing loop

### Action categoricals

The code contains a specific categorical distribution type for reaction actions, `ActionCategorical`. This class contains logic to sample from concatenated sets of logits accross a minibatch. 

The `ActionCategorical` class handles this and can be used to compute various other things, such log probabilities; it can also be used to sample from the distribution.

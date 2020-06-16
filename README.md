# sparsify [WIP]
A simple, flexible, no-nonsense neural network sparsification library for PyTorch

## What ? 
Sparsify is a pruning library that aims to make research in the field of pruning, model compression, 
sparse neural networks easy. You can take your PyTorch model and replace conv, dense layers with 
masked layers. The pruner class has everything you need for weight pruning, but can easily be 
extended for other types of importance measures. 

## Currently supported: 
* Weight and unit pruning
    * Global and per-layer pruning 
    * One time and Ramping pruning 
    * Weight, mask reset
    
## To-do: 
* Implement Single shot pruning algorithms like SNIP, GraSP 
* Have a generalized pruner class that can take in any pytorch model instead of replacing 
layers for a given model (use hooks to override forward method)
* Extend support for Language models, Transformers 
* Write unit-tests, examples 

### Code Credits 
These repositories have been helpful.  
* [Some help in pruning methods](https://github.com/srk97/defense)
* [Vision models](https://github.com/pytorch/vision/tree/master/torchvision/models)

### Contributions
Please feel free to suggest ways to improve or add new features, open an issue for the same. 


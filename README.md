# Gradient Assisted Learning
This is an implementation of Gradient Assisted Learning
 
## Requirements
See requirements.txt

## Instruction
 - Global hyperparameters are configured in config.yml
 - Experimental setup are listed in make.py 
 - Hyperparameters can be found at process_control() in utils.py 
 - organization.py define local initialization, learning, and inference of one organization
 - assist.py demonstrate Gradient Assisted Learning algorithm
    - broadcast() compute and distribute the pseudo-residual to all organizations
    - update() gather other organizations' output and compute gradient assisted learning rate and gradient assistance weights
 - The features are split at split_dataset() in data.py and apply at feature_split() in models/utils.py

## Examples
 - Run make_stats.py to make normalization statistics for each dataset
 - Train GAL for CIFAR10 dataset (CNN, <img src="https://latex.codecogs.com/gif.latex?M=8"/>, with gradient assistance weights, <img src="https://latex.codecogs.com/gif.latex?E=10"/>, <img src="https://latex.codecogs.com/gif.latex?T=10"/>, with gradient assisted learning rate, noise free)
    ```ruby
    python train_model_assist.py --data_name CIFAR10 --model_name conv --control_name 8_stack_10_10_search_0
    ```
 - Test GAL for ModelNet40 dataset (CNN, <img src="https://latex.codecogs.com/gif.latex?M=12"/>, no gradient assistance weights, <img src="https://latex.codecogs.com/gif.latex?E=10"/>, <img src="https://latex.codecogs.com/gif.latex?T=10"/>, no gradient assisted learning rate, <img src="https://latex.codecogs.com/gif.latex?\sigma=1"/>)
    ```ruby
    python test_model_assist.py --data_name ModelNet40 --model_name conv --control_name 12_bag_10_10_fix_1
    ```

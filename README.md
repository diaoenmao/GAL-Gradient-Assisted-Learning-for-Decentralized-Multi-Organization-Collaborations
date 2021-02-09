# Gradient Assisted Learning
This is an implementation of Gradient Assisted Learning
 
## Requirements
See requirements.txt

## Instruction

 - Global hyperparameters are configured in config.yml
 - Hyperparameters can be found at process_control() in utils.py 
 - organization.py define local initialization, learning, and inference of one organization
 - assist.py demonstrate Gradient Assisted Learning algorithm, broadcast() compute and distribute the pseudo-residual to all organizations
 - The features are split at split_dataset() in data.py and apply at feature_split() in models/utils.py

## Details
 - Run make_stats.py to make normalization statistics for each dataset
 - Train GAL-b for QSAR dataset with linear model, M=4, E=100 and T=10
    ```ruby
    python train_model_assist.py --data_name QSAR --model_name linear --control_name 4_bag_100_10
    ```
 - Train GAL-s for MNIST dataset with CNN model, M=8, E=10 and T=10
    ```ruby
    python train_model_assist.py --data_name MNIST --model_name conv --control_name 8_stack_10_10
    ```
 - Test GAL-s for CIFAR10 dataset with CNN model, M=8, E=10 and T=10
    ```ruby
    python test_model_assist.py --data_name CIFAR10 --model_name conv --control_name 8_stack_10_10
    ```
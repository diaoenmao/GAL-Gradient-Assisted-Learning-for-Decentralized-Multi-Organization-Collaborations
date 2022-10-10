# GAL: Gradient Assisted Learning for Decentralized Multi-Organization Collaborations
This is an implementation of [GAL: Gradient Assisted Learning for Decentralized Multi-Organization Collaborations](https://arxiv.org/abs/2106.01425)
- Decentralized organizations form a community of shared interest to provide better Machine-Learning-as-a-Service.

<img src="/asset/AL.png">

- Learning and Prediction Stages for Gradient Assisted Learning (GAL).

<img src="/asset/GAL.png">

## Requirements
See requirements.txt

## Instructions
 - Global hyperparameters are configured in config.yml
 - Use make.py to generate exp script
 - Use process.py to process exp results
 - Hyperparameters can be found in config.yml and process_control() in utils.py
 - organization.py define local initialization, learning, and inference of one organization
 - assist.py demonstrate Gradient Assisted Learning algorithm
    - broadcast() compute and distribute the pseudo-residual to all organizations
    - update() gather other organizations' output and compute gradient assisted learning rate and gradient assistance weights
 - The features are split at split_dataset() in data.py and apply at feature_split() in models/utils.py

## Examples
 - Run make_stats.py to make normalization statistics for each dataset
 - Train 'Joint' for Wine dataset (Linear, $M=1$, no gradient assistance weights, $E=100$, $T=10$, with gradient assisted learning rate, noise free)
    ```ruby
    python train_model_assist.py --data_name Wine --model_name linear --control_name 1_none_100_10_search_0
    ```
 - Test 'Alone' for QSAR dataset (Linear, $M=8$, no gradient assistance weights, $E=100$, $T=10$, with gradient assisted learning rate, noise free)
    ```ruby
    python test_model_assist.py --data_name QSAR --model_name linear --control_name 8_none_100_10_search_0
    ```
 - Train GAL for CIFAR10 dataset (CNN, $M=8$, with gradient assistance weights, $E=10$, $T=10$, with gradient assisted learning rate, noise free)
    ```ruby
    python train_model_assist.py --data_name CIFAR10 --model_name conv --control_name 8_stack_10_10_search_0
    ```
 - Test GAL for ModelNet40 dataset (CNN, $M=12$, no gradient assistance weights, $E=10$, $T=10$, no gradient assisted learning rate, $\sigma=1$)
    ```ruby
    python test_model_assist.py --data_name ModelNet40 --model_name conv --control_name 12_bag_10_10_fix_1
    ```
    
## Results
- Results of the UCI datasets ($M=8$) with Linear, GB, SVM and GB-SVM models. The Diabetes and Boston Housing (regression) are evaluated with Mean Absolute Deviation (MAD), and the rest (classification) are evaluated with Accuracy.

| Dataset |  Model | Diabetes$(\downarrow)$ | BostonHousing$(\downarrow)$ | Blob$(\uparrow)$ | Wine$(\uparrow)$ | BreastCancer$(\uparrow)$ | QSAR$(\uparrow)$ |
|:-------:|:------:|:----------------------:|:---------------------------:|:----------------:|:----------------:|:------------------------:|:----------------:|
|   Late  | Linear |       136.2(0.1)       |           8.0(0.0)          |    100.0(0.0)    |    100.0(0.0)    |         96.9(0.4)        |     76.9(0.8)    |
|  Joint  | Linear |        43.4(0.3)       |           3.0(0.0)          |    100.0(0.0)    |    100.0(0.0)    |         98.9(0.4)        |     84.0(0.2)    |
|  Alone  | Linear |        59.7(9.2)       |           5.8(0.9)          |    41.3(10.8)    |    63.9(15.6)    |         92.5(3.4)        |     68.8(3.4)    |
|    AL   | Linear |        51.5(4.6)       |           4.7(0.6)          |     97.5(2.5)    |     95.1(3.6)    |         97.7(1.1)        |     70.6(5.2)    |
|   GAL   | Linear |        42.7(0.6)       |           3.2(0.2)          |    100.0(0.0)    |     96.5(3.0)    |         98.5(0.7)        |     82.5(0.8)    |
|   GAL   |   GB   |        56.5(2.8)       |           3.8(0.5)          |     96.3(2.2)    |     95.8(1.4)    |         96.1(1.0)        |     84.8(0.9)    |
|   GAL   |   SVM  |        46.6(1.4)       |           2.9(0.2)          |     96.3(4.1)    |     96.5(1.2)    |         99.1(1.1)        |     85.5(0.7)    |
|   GAL   | GB-SVM |        49.8(2.6)       |           3.4(0.8)          |     70.0(7.9)    |     95.8(1.4)    |         93.2(1.6)        |     82.9(1.5)    |

- Results of the CIFAR10 (a-c) ($M=8$) and MIMICL (d-f) ($M=4$) datasets. GAL significantly outperforms 'Alone' and 'AL'.

![MNIST_interp_iid](/asset/CIFAR10_8_MIMICL_4_assist.png)

## Acknowledgements
*Enmao Diao  
Jie Ding  
Vahid Tarokh*
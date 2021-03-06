# README of Group 01

## Dependency and Environment settings
  ```
  Package                  Version  
  ------------------------ ---------
  absl-py                  0.10.0   
  astor                    0.8.1    
  attrs                    20.2.0   
  certifi                  2020.6.20
  chardet                  3.0.4    
  decorator                4.4.2    
  dill                     0.3.2    
  future                   0.18.2   
  gast                     0.2.2    
  google-pasta             0.2.0    
  googleapis-common-protos 1.52.0   
  grpcio                   1.33.1   
  h5py                     2.10.0   
  idna                     2.10     
  importlib-metadata       2.0.0    
  Keras-Applications       1.0.8    
  Keras-Preprocessing      1.1.2    
  Markdown                 3.3.3    
  networkx                 2.5      
  numpy                    1.19.2   
  opt-einsum               3.3.0    
  pip                      19.2.3   
  promise                  2.3      
  protobuf                 3.13.0   
  requests                 2.24.0   
  scipy                    1.5.3    
  setuptools               41.2.0   
  six                      1.15.0   
  tensorboard              1.15.0   
  tensorflow               1.15.0         # If no GPU is available
  tensorflow-datasets      1.3.2          # Important
  tensorflow-estimator     1.15.1   
  tensorflow-gpu           1.15.0         # Important  
  tensorflow-metadata      0.24.0   
  termcolor                1.1.0    
  tqdm                     4.51.0   
  urllib3                  1.25.11  
  Werkzeug                 1.0.1    
  wheel                    0.35.1   
  wrapt                    1.12.1   
  zipp                     3.4.0
  ```

  On top of that, we use Python 3.7.6 for our experiments.
  Most version of package is not relevant. The two most important packages are
  tensorflow-gpu (or tensorflow if no gpu is available) and
  tensorflow-datasets. We require tensorflow-datasets==1.3.2, and
  tensorflow-gpu==1.15.0 (or tensorflow==1.15.0).

  We ran our experiment in Linux, with release `Ubuntu 18.04.4 LTS`.
  Specifically, `4.15.0-112-generic #113-Ubuntu SMP Thu Jul 9 23:41:39 UTC 2020
  x86_64 x86_64 x86_64 GNU/Linux`.

  All experiments are conducted in a single GeForce RTX 2080 Ti GPU.

## How to run experiments

  The log files generated by scripts may overwrite each other, so make sure you
  save log files before running new scripts. Or just prepare a new project
  folder for that.

  * Main results (GCN 16):

    With correct dependency and settings, you can obtain the main results in
    the report by simply executing the script,
    ```
    chmod +x run_experiment.sh
    ./run_experiment.sh
    ```

    which will generate log files under folder
    `log_${noise_level}/${step}/${seed}/`, with format
    `${dataset}_${model}_reg-${reg_lu}.log`. For example,
    `log_0.3/30/1234/cora_gcn_16_reg-100.log`.

  * (Optional) One co-training step with GCN 16:

    Similarly,
    ```
    chmod +x run_experiment_step-1.sh
    ./run_experiment_step-1.sh
    ```

  * (Optional) Hyperparameter search for lambda

    Similarly,
    ```
    chmod +x check_lambda.sh
    ./check_lambda.sh
    ```

  * (Optional) MLP 128 and GCN 128 results:

    Similarly,
    ```
    chmod +x run_experiment_hidden_128.sh
    chmod +x run_experiment_pubmed-full.sh
    ./run_experiment_hidden_128.sh
    ./run_experiment_pubmed-full.sh
    ```

    The format is a bit different, with `${step}` folder omitted, i.e.
    it will generate log files under folder `log_${noise_level}/${seed}/`, with
    format `${dataset}_${model}_reg-${reg_lu}.log`. For example,
    `log_0.3/1234/cora_gcn_16_reg-100.log`. For
    `run_experiment_pubmed-full.sh`, it is under folder
    `pubmed_full_log_${noise_level}/${seed}/`.

## Files

  ```
  |- data_{noise_level}      # IMPORTANT: Data with different noise level
  |- README.md               # README file
  |- .gitignore              # .gitignore file
  |- *.png                   # (Irrelevant) Original code's figures
  |- run_configs.txt         # (Irrelevant) Original paper's running configurations
  |- *.sh                    # IMPORTANT: scripts of our experiments
  |- summary*.log            # IMPORTANT: final result of our experiments
  |- gam                     # IMPORTANT: source code
                             #     Logics: scripts -> experiments -> trainer
    |- experiments           #   source code: 'main.py' for running experiments
    |- trainer               #   source code: training processes
    |- models                #   source code: model graph implemented in TF
    |- data                  #   source code: dataset classes
  ```

## Experiment results in the report
  * Main results (GCN 16): `summary_step-30.log`

    You can use following commands to check statistics:
    ```
    cat summary_step-30.log | awk 'NF == 7 { sum_sqr[$1" "$3" "$4] += $7 * $7; sum[$1" "$3" "$4] += $7; count[$1" "$3" "$4] += 1.0; } END { for (key in count) { mean=sum[key] / count[key]; variance=sum_sqr[key] / count[key] - mean * mean; stddev = sqrt(variance); print key" "mean * 100" $\\pm$ "stddev * 100; } }' | sort
    ```

    Instead, if you want to generate this file yourself from experiment logs,
    you can run the following command:
    ```
    for step in 30; do for noise in log_0 log_0.3 log_0.5; do for seed in 1234 2345 5130 7840 9997; do for dataset in cora citeseer pubmed; do for model in  gcn_16; do for reg in  100; do file=${noise}/${step}/${seed}/${dataset}_${model}_reg-${reg}.log; echo "${noise} ${seed} ${dataset} ${model} ${reg} ${file} $(tail -2 ${file} | head -1 | sed 's/.*test acc: \(.*\) at.*/\1/')"; done; done; done; done; done; done > summary_step-30.log
    ```

  * (Optional) One co-training step with GCN 16: `summary_step-1.log`

    You can use following commands to check statistics:
    ```
    cat summary_step-1.log | awk 'NF == 7 { sum_sqr[$1" "$3" "$4] += $7 * $7; sum[$1" "$3" "$4] += $7; count[$1" "$3" "$4] += 1.0; } END { for (key in count) { mean=sum[key] / count[key]; variance=sum_sqr[key] / count[key] - mean * mean; stddev = sqrt(variance); print key" "mean * 100" $\\pm$ "stddev * 100; } }' | sort
    ```

    Similarly, you can generate this file by yourself from experiment logs,
    ```
    for step in 1; do for noise in log_0 log_0.3 log_0.5; do for seed in 1234 2345 9999 7839 5129 3584 4976 2360 1112 7713; do for dataset in cora citeseer pubmed; do for model in  gcn_16; do for reg in  100; do file=${noise}/${step}/${seed}/${dataset}_${model}_reg-${reg}.log; echo "${noise} ${seed} ${dataset} ${model} ${reg} ${file} $(tail -2 ${file} | head -1 | sed 's/.*test acc: \(.*\) at.*/\1/')"; done; done; done; done; done; done > summary_step-1.log
    ```

  * (Optional) MLP 128 and GCN 128 results: `summary_hidden_128.log` and `summary_pubmed-full.log`

    The MLP 128 results can be checked via command
    ```
     cat summary_hidden_128.log | awk 'NF == 6 { sum_sqr[$1" "$3" "$4] += $6 * $6; sum[$1" "$3" "$4] += $6; count[$1" "$3" "$4] += 1.0; } END { for (key in count) { mean=sum[key] / count[key]; variance=sum_sqr[key] / count[key] - mean * mean; stddev = sqrt(variance); print key" "mean * 100" $\\pm$ "stddev * 100; } }' | grep mlp_128 | sort
    ```

    The GCN 128 reproduced results can be viewed with following commands,
    ```
    # Cora and Citeseer
    cat summary_hidden_128.log | grep 1234 | grep gcn | grep "log_0 " | grep -v "pubmed"

    # Pubmed
    cat summary_pubmed-full.log | grep 1234 | grep gcn | grep "log_0 " | grep "pubmed"
    ```

    The `summary_hidden_128.log` file can be generated from experiment logs,
    ```
    for noise in log_0 log_0.3 log_0.5; do  for dataset in cora citeseer pubmed; do for model in mlp_128 mlp_32_32_32_32 gcn_128; do for seed in 1234 2345 9999 7839 5129 3584 4976 2360 1112 7713; do file=${noise}/${seed}/${dataset}_${model}.log; echo "${noise} ${seed} ${dataset} ${model} ${file} $(tail -2 ${file} | head -1 | sed 's/.*test acc: \(.*\) at iteration.*/\1/')"; done; done; done; done > summary_hidden_128.log
    ```

    The `summary_pubmed-full.log` file can be generated from experiment logs,
    ```
    for noise in pubmed_full_log_0 pubmed_full_log_0.3 pubmed_full_log_0.5; do  for dataset in pubmed; do for model in  gcn_128; do for seed in 1234 2345 9999 7839 5129; do file=${noise}/${seed}/${dataset}_${model}.log; echo "${noise} ${seed} ${dataset} ${model} ${file} $(tail -2 ${file} | head -1 | sed 's/.*test acc: \(.*\) at iteration.*/\1/')"; done; done; done; done > summary_pubmed-full.log
    ```


# GAM: Graph Agreement Models for Semi-Supervised Learning

This code repository contains an implementation of Graph Agreement Models [1].

Neural structured learning methods such as Neural Graph Machines [2], Graph
Convolutional Networks [3] and their variants have successfully combined the
expressiveness of neural networks with graph structures to improve on learning
tasks. Graph Agreement Models (GAM) is a technique that can be applied to these
methods to handle the noisy nature of real-world graphs. Traditional graph-based
algorithms, such as label propagation, were designed with the underlying
assumption that the label of a node can be imputed from that of the neighboring
nodes and edge weights. However, most real-world graphs are either noisy or have
edges that do not correspond to label agreement uniformly across the graph.
Graph Agreement Models introduce an auxiliary model that predicts the
probability of two nodes sharing the same label as a learned function of their
features. This agreement model is then used when training a node classification
model by encouraging agreement only for those pairs of nodes that it deems
likely to have the same label, thus guiding its parameters to a better local
optima. The classification and agreement models are trained jointly in a
co-training fashion.

The code is organized into the following folders:

*   data: Classes and methods for accessing semi-supervised learning datasets.
*   models: Classes and methods for classification models and graph agreement
    models.
*   trainer: Classes and methods for training the classification models, and
    agreement models individually as well as in a co-training fashion.
*   experiments: Python run script for training Graph Agreement Models on
    CIFAR10 and other datasets.

The implementations of Graph Agreement Models (GAMs) are provided in the `gam`
folder on a strict "as is" basis, without warranties or conditions of any kind.
Also, these implementations may not be compatible with certain TensorFlow
versions (such as 2.0 or above) or Python versions.

More details can be found in our
[paper](https://papers.nips.cc/paper/9076-graph-agreement-models-for-semi-supervised-learning.pdf),
[supplementary material](https://papers.nips.cc/paper/9076-graph-agreement-models-for-semi-supervised-learning-supplemental.zip),
[slides](https://drive.google.com/open?id=1tWEMoyrbLnzfSfTfYFi9eWgZWaPKF3Uu) or
[poster](https://drive.google.com/file/d/1BZNR4B-xM41hdLLqx4mLsQ4KKJOhjgqV/view).

## How to run

To run GAM on a graph-based dataset (e.g., Cora, Citeseer, Pubmed), from this
folder run: `$ python3.7 -m gam.experiments.run_train_gam_graph
--data_path=<path_to_data>`

To run GAM on datasets without a graph (e.g., CIFAR10), from this folder run: `$
python3.7 -m gam.experiments.run_train_gam`

We recommend running on a GPU. With CUDA, this can be done by prepending
`CUDA_VISIBLE_DEVICES=<your-gpu-number>` in front of the run command.

For running on different datasets and configuration, please check the command
line flags in each of the run scripts. The configurations used in our paper can
be found in the file `run_configs.txt`.

## Visualizing the results.

To visualize the results in Tensorboard, use the following command, adjusting
the dataset name accordingly: `$ tensorboard --logdir=outputs/summaries/cora`

An example of such visualization for Cora with GCN + GAM model on the Pubmed
dataset is the following:
![Tensorboard plot](gam_gcn_pubmed.png?raw=true "GCN + GAM on Pubmed")

Similarly, we can run with multiple different parameter configurations and plot
the results together for comparison. An example showing the accuracy per
co-train iteration of a GCN + GAM model on the Cora dataset for 3 runs with 3
different random seeds is the following:
![Tensorboard plot](gam_gcn_cora_multiple_seeds.png?raw=true "GCN + GAM on Cora")

## References

[[1] O. Stretcu, K. Viswanathan, D. Movshovitz-Attias, E.A. Platanios, S. Ravi,
A. Tomkins. "Graph Agreement Models for Semi-Supervised Learning." NeurIPS
2019](https://papers.nips.cc/paper/9076-graph-agreement-models-for-semi-supervised-learning)

[[2] T. Bui, S. Ravi and V. Ramavajjala. "Neural Graph Learning: Training Neural
Networks Using Graphs." WSDM 2018](https://research.google/pubs/pub46568.pdf)

[[3] T. Kipf and M. Welling. "Semi-supervised classification with graph
convolutional networks." ICLR 2017](https://arxiv.org/pdf/1609.02907.pdf)

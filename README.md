# About
This repository contains the codes and related materials for the paper `Learning Distributionally Robust Tractable Probabilistic Models in Continuous Domains` appears on UAI 2024.

In this work, we explore the possibility of applying Distritbutionally Robust Supervised Learning (DRSL) to learn generative probabilistic graphic models and tackling the adversarial risk minimization problem within the framework of distributionally robust learning.

We develop highly efficient algorithms and demonstrate that the adversarial risk minimization problem can be efficiently addressed when the model permits exact log-likelihood evaluation and efficient learning on weighted data. 


# Getting Started

## Dataset 
All of the dataset used in the paper can be downloaded [here][data_url]. 

After downloading the data to your local machine, move the file to the project directory, and extract the data using command `tar -xzf uai2024-robust-data.tar.gz`

## Run the Experiment
> You might need to install the dependencies first, the file `requirements.txt` lists all packages used by ours, but only part of them are required for running this project.

> You also need to add the project directory to the python path (e.g. the Linux command `export PYTHONPATH=$(pwd):$PYTHONPATH`). 

The main script we used in our experiments are jupyter notebooks, and we use `papermill` to run the notebook with input arguments:
- data_name: the name of the dataset 
- testing: boolean flag, 1 means use testing mode, where the number of data, iterations, and hyperparameter combinations will be reduced for quick debugging.
- log_file: the log_file path
- G_delta: the value of `delta` parameter

Check `run.sh` for an example of running the notebooks. Note that you can always open the notebook and run them manually as need.

We also provide `parallel.sh` for running multiple experiments simultaneously in parallel.

## Adapt for Other Models
If you wish to adapt our algorithm for DRSL to learn other probabilistic models, you can check the two main jupyter notebooks `mixmg_main.ipynb` and `nngbn_main.ipynb` for details.

The core function you will need is `adversarial_step` defined in both notebooks.


# Results
We evaluated the generative performance of the robust model (trained through DRSL) against its standard counterparts (trained through MLE) for both Mixture of Multivariate Gaussian (MixMG) and [NN-GBN][cont_cnet] models.

The average loglikelihoods gain achieved by DRSL compared against MLE are shown in the following figure (first one is original test set while the other four are adversarial test sets).

![experiment results][result_fig]



# Citation
Please cite our work if you find it is helpful for your research! 

(The bibtex will be updated later.)


# Contact
If you have any questions or need help regarding our work, you can email us and we are happy to discuss the work (the email addresses of each author are included as hyperlinks in the paper). 

In case my school email being deactivated, you can email me using my personal email address `HailiangDong@hotmail.com`.


[data_url]: https://utdallas.box.com/s/4wnf7mk5c0jj49vvibjxuyk7su8lmxij
[result_fig]: https://github.com/LeonDong1993/UAI2024-RobustLearning/blob/main/figs/ll_improvement.png
[cont_cnet]: https://proceedings.mlr.press/v151/dong22a/dong22a.pdf

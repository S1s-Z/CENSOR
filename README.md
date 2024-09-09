# CENSOR

Hi, this is the code of our paper "Improving the Robustness of Distantly-Supervised Named Entity Recognition via Uncertainty-Aware Teacher Learning and Student-Student Collaborative Learning" accepted by ACL 2024 Findings. Our paper is available [here](https://aclanthology.org/2024.findings-acl.329/).

## News:

Accepted by ACL 2023 Findings. 2023.05

Code released at Github. 2023.08

## Preparation

Use requirements.txt to get the right environments

## Reproduce results

We provide three bash scripts run_conll03.sh, run_ontonotes5.sh, run_webpage.sh, run_wikigold.sh, run_twitter.sh for running the model on the datasets.

you can run the code like:

```
 sh <run_dataset>.sh <GPU ID> <DATASET NAME>
```


e.g.

```
sh run_conll03.sh 0,1 conll03
```


We got our results in single A40.

# ml-project-2-orlovietf

# Protein Melting Temperature Prediction

<p align="center">
  <a href="#about">About</a> •
  <a href="#tutorials">Tutorials</a> •
  <a href="#examples">Examples</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

<p align="center">
<a href="https://github.com/Blinorot/pytorch_project_template/blob/main/LICENSE">
   <img src=https://img.shields.io/badge/license-MIT-blue.svg>
</a>
</p>

## About

This repository contains the work done for our second [Machine Learning course](https://www.epfl.ch/labs/mlo/machine-learning-cs-433/) project, which was completed in conjunction with the [Laboratory for Biomolecular Modelling](https://www.epfl.ch/labs/lbm/) under the mentorship of [Lucien Krapp](https://people.epfl.ch/lucien.krapp). The goal of our effort is to use amino acid sequences to predict the melting temperature (Tm) of proteins, which an important characteristic that indicates a protein's thermal stability.

## Data

The initial approach was to use the [train data](https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/data?select=train.csv) publicly available for Kaggle's competition [Novozymes Enzyme Stability Prediction](https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/data). There were updates on this file made by the competition organizers and the final train file used is [train_updated](https://github.com/CS-433/ml-project-2-orlovietf/blob/main/train_updated.csv).

One part of the experiments was conducted by directly using sequences as inputs to our methods and values of tm as outputs. However, based on this [discussion](https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/discussion/358320), we incorporated the knowledge based data preprocessing to obtain [TO-DO].

## Installation [TO-DO]

Installation may depend on your task. The general steps are the following:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use [TO-DO]

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

## Method

We implemented two main approaches for this problem.

- Pretrained ESM 2 Model: Utilize the pretrained [ESM 2 model](https://huggingface.co/docs/transformers/model_doc/esm) from Hugging Face to generate sequence embeddings. These embeddings are further fine-tuned by training a [neural network](https://github.com/CS-433/ml-project-2-orlovietf/blob/main/scripts/baseline_model.py) to predict Tm values.

- Carbonara Architecture: Used the [Carbonara architecture](https://github.com/LBM-EPFL/CARBonAra/tree/main) embeddings from our data (precisely, the output from the penultimate layer) and:
    1. Obtained features used to train a neural network.
    2. Used RNN model on embeddings directly.

## Credits

This repository is based on a heavily modified fork of [pytorch-template](https://github.com/victoresque/pytorch-template) and [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repositories.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
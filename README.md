# Reducing Batch Effects in Bio-Image Analysis using Adversarial Learning on RxRx1

This is a collaborative open-source research project focused on developing robust models for biological image analysis by addressing batch effects using adversarial training.

## Overview

Batch effects are systematic variations introduced during biological imaging that can negatively impact model performance and generalization. This project uses Domain Adversarial Neural Networks (DANN) to reduce the influence of batch effects and enable more reliable classification of cellular images.

## Motivation

Batch effects are a well-known issue in biomedical image analysis. They can lead models to learn technical noise rather than biologically meaningful patterns. By using adversarial learning, we aim to train models that are invariant to experimental conditions (like the imaging plate), leading to better performance in real-world applications.

This work supports:

* More reliable drug discovery pipelines
* Better cross-laboratory generalization
* Improved diagnostic tools
* A stronger foundation for future biomedical datasets

## Problem Statement

Build a model that classifies biological signals from microscopy images while ignoring batch-specific signals such as the imaging plate. This is done by:

* Training a feature extractor on image data
* Predicting the biological class (e.g., cell type)
* Simultaneously training a domain discriminator to predict the plate ID — with a gradient reversal layer so the feature extractor learns to ignore it

The end result is a plate-invariant model that generalizes better across batches.

## Dataset

We use the [RxRx1 dataset from Recursion Pharmaceuticals](https://www.kaggle.com/competitions/recursion-cellular-image-classification). It contains:

* Images of human cells treated with various siRNA
* Data collected from 51 experimental plates
* 6-channel fluorescence images
* Naturally occurring batch effects, making it a perfect fit for this problem

## Approach

We use Domain Adversarial Neural Networks (DANN), where:

* The model learns features from the input images
* A label predictor learns to classify cell types or perturbations
* A domain classifier (adversary) tries to identify which plate the image came from
* A gradient reversal layer ensures the feature extractor learns features useful for the label but not for the domain

This adversarial setup makes the model blind to batch effects and better at generalizing across different plates.

## Baselines

We are currently implementing and testing baseline models that:

* Use standard convolutional architectures
* Evaluate performance with and without adversarial training
* Visualize batch separation using t-SNE and other dimensionality reduction techniques

Results will be shared as the project progresses.

## Project Structure
```
project-A/
├── data/                    # raw & processed data
│   ├── raw/
│   ├── processed/
├── notebooks/              # exploratory analysis
├── src/
│   ├── data/               # data loaders, augmentation
│   ├── models/             # adversarial networks
│   ├── training/           # training scripts, loss functions
│   ├── evaluation/         # evaluation metrics, plotting
├── configs/                # experiment configs (Hydra/YAML)
├── scripts/                # run scripts (train.py, test.py)
├── results/                # logs, model checkpoints
├── requirements.txt
├── README.md
└── .gitignore
```

## Kaggle Notebook

We are actively working on a Kaggle notebook for:

* Dataset exploration
* Model prototyping
* Baseline comparison
* Visualizations and debugging

The final notebook link will be shared here. Team members are encouraged to upload their own notebooks in the `/notebooks/` directory on GitHub, with filenames that include their initials or name.

## Team
* **Divyansh Goyal** ([GitHub](https://github.com/divital-coder))
* **Mridul Jain** ([GitHub](https://github.com/Spinachboul))
* **Kyung Seo Kim** ([GitHub](https://github.com/kkyungseo))
* **Showrin Rahman**[GitHub](https://github.com/showrin20)

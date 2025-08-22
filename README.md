# neuralif

This repository contains the code for learning incomplete factorization preconditioners, modified for my masters thesis by Guy McClennan.

The initial codebase was based on the repository of [Paul Häusner](https://paulhausner.github.io), Aleix Nieto Juscafresa, [Ozan Öktem](https://www.kth.se/profile/ozan), and [Jens Sjölund](https://jsjol.github.io/) for their paper on Neural Incomplete Factorization.

Much of the initial code had to be rewritten due to problems with package dependencies.

## Installation

In order to run the training and testing, you need to install the following python dependencies:

- pytorch
- pytorch-geometric (and its dependencies like torch-scatter and torch-sparse)
- scipy
- matplotlib

## Implementation

The repository consists of several parts. In the `krylov` folder implementations for the conjugate gradient method.

The `neuralif` module contains the code for the learned preconditioner. The model.py file contains the different models that can be utilizes, loss.py implements several different loss functions.

A synthetic dataset is provided in the folder `apps`.

The preprocess_data.py script takes your raw dataset of matrices and converts each matrix into a graph representation, calculates all the necessary node features (like degree, diagonal dominance, etc.) for every node in every graph, packages the graph structure, node features, and edge features into a single, optimized file for each matrix and saves these optimised files to a new directory (e.g., data/processed/), which will be the source for the training script. This is needed for the `NeuralIF-K` models.

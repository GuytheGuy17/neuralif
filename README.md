# Neural incomplete factorization

This repository contains the code for learning incomplete factorization preconditioners directly from data by [Paul Häusner](https://paulhausner.github.io), Aleix Nieto Juscafresa, [Ozan Öktem](https://www.kth.se/profile/ozan), and [Jens Sjölund](https://jsjol.github.io/).

## Installation

In order to run the training and testing, you need to install the following python dependencies:

- pytorch
- pytorch-geometric
- scipy
- networkx

For validation and testing the following packages are required:

- matplotlib
- [numml](https://github.com/nicknytko/numml) (for efficient forward-backward substitution)
- [ilupp](https://github.com/c-f-h/ilupp) (for baseline incomplete factorization preconditioners)

## Implementation

The repository consists of several parts. In the `krylov` folder implementations for the conjugate gradient method and GRMES method are provided. Further, several preconditioner (Jacobi, ILU, IC) are implemented.

The `neuralif` module contains the code for the learned preconditioner. The model.py file contains the different models that can be utilizes, loss.py implements several different loss functions.

A synthetic dataset is provided in the folder `apps`.

## References

If our code helps your research or work, please consider citing our paper. The following are BibTeX references:

```
@article{hausner2024neural,
  title={Neural incomplete factorization: learning preconditioners for the conjugate gradient method},
  author={Paul H{\"a}usner and Ozan {\"O}ktem and Jens Sj{\"o}lund},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2024},
  url={https://openreview.net/forum?id=FozLrZ3CI5}
}

@InProceedings{hausner2025learning,
  title={Learning incomplete factorization preconditioners for {GMRES}},
  author={H{\"a}usner, Paul and Nieto Juscafresa, Aleix and Sj{\"o}lund, Jens},
  booktitle={Proceedings of the 6th Northern Lights Deep Learning Conference (NLDL)},
  pages={85--99},
  year={2025},
  volume={265},
  series={Proceedings of Machine Learning Research},
  publisher={PMLR},
}

```

Please feel free to reach out if you have any questions or comments.

Contact: Paul Häusner, paul.hausner@it.uu.se

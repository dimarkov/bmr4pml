# Bayesian Model Reduction for Probabilistic Machine Learning (bmr4pml)

This repository contains the code accompanying our paper on stochastic Bayesian Model Reduction for efficient sparsification of deep neural networks. Our approach leverages stochastic and black-box variational inference and Bayesian model reduction, a generalization of the Savage-Dickey ratio. The stochastic BMR approach allows for an iterative pruning of model parameters based on posterior estimates obtained from a simple variational mean-field approximation of the generative model with Gaussian priors over model parameters and layer-specific scale parameters. This leads to a computationally efficient pruning algorithm where the pruning step is computationally negligible in comparison to SVI optimization. While the initial focus is on image classification, we have plans to expand the scope of this project to cover various modern probabilistic machine learning tasks.

We welcome [contributions](#contributing) from the community to help us extend the capabilities of bmr4pml to other areas of probabilistic machine learning. If you have ideas, code improvements, or specific use cases in mind, we encourage you to get involved. Whether it's adding support for new datasets, implementing different network architectures, or improving the algorithm's efficiency, your contributions can help make this project even more versatile and valuable.

## Table of Contents
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

### Prerequisites

Before installing **bmr4pml**, ensure you have the following prerequisites installed:

- **Python:** You'll need Python 3.10 or higher.

- **JAX with CUDA Support (Optional):** **bmr4pml** leverages JAX, a numerical computing library. CUDA support in JAX allows for GPU acceleration, but it's optional and depends on your hardware configuration. You can install JAX with CUDA support manually by following the installation instructions at the official JAX [repository](https://github.com/google/jax).

### Installation
You can install the project using either of the following methods:

#### Option 1: Install via Pip

```shell
$ pip install bmr4pml 
```
This will install the latest stable release of the project and its dependencies.

#### Option 2: Developer's Version

If you want to work with the development version or make contributions, follow these steps:

```shell
$ git clone https://github.com/yourusername/bmr4pml.git
$ cd bmr4pml
$ pip install -e .
```
This will install the project locally, allowing you to make changes and have them immediately reflected without the need for reinstalling. It also installs the development dependencies.

Choose the installation option that best suits your needs.

## Usage
Learn how to use our project by checking out the examples provided in the 'notebooks' and 'scripts' directories.

## Contributing
We welcome contributions from the community. If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature: git checkout -b feature-name.
3. Make your changes and commit them: git commit -m 'Description of changes'.
4. Push to your fork: git push origin feature-name.
5. Create a pull request on the original repository's branch.
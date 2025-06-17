# Conditional Bayesian Flow Networks

This repository is based on the official code release for [Bayesian Flow Networks](https://arxiv.org/abs/2308.07037) by Alex Graves, Rupesh Kumar Srivastava, Timothy Atkinson and Faustino Gomez. We made some improvements and adjustments to enable the conditional generation.

## Reading Guide

- `checkpoints/` contains the archieved files of our experimental results of the conditional BFN, including models and training dynamics.
- `configs/` contains the configs of the BFN model, default to `mnist_discrete_.yaml` in this repository.
- `flow_visualization/` contains three sets of samples drawn from the generating process of the output distribution and input distribution.
- `networks/` contains implementations of the network architectures used by the models. 
- `data.py` contains utilities related to data loading and processing.
- `model.py` contains all the main contributions of the original paper and our modifications. These include definitions for discrete data, of Bayesian Flows as well as loss functions for both continuous-time and discrete-time. See comments in the base classes in that file for details.
- `probability.py` defines the probability distributions used by the models.
- `train.py`, `test.py` and `sample.py` are scripts for training, testing and sampling (see below for usage).
- `visualize_flow.py` is the script for visualizing the generating process for some given condition (0-9 on MNIST in our code).

## Setup

```shell
# Create a new conda env with all dependencies including pytorch and CUDA
conda env create -f env.yml
conda activate bfn

# Or, install additional dependencies into an existing pytorch env
pip install accelerate==0.19.0 matplotlib omegaconf rich

# Optional, if you want to enable logging to neptune.ai
pip install neptune 
```

## Training

The models in the paper can be trained using the configs provided in the `configs` dir as follows:

```shell
# mnist experiment on 1 GPU
accelerate launch train.py
```

## Testing
> [!NOTE]
> Depending on your GPU, you may wish to adjust the batch size used for testing in `test.py`.
```shell
# Compute 784-step loss on MNIST
python test.py seed=1 config_file=./configs/mnist_discrete_.yaml load_model= n_steps=784 n_repeats=2000
```
> [!IMPORTANT]
> All computed results will be in nats-per-data-dimension. To convert to bits, divide by $\ln(2)$.

## Sampling

You can sample from a pre-trained model as follows (change options as desired):

```shell
# Sample 4 binarized MNIST images using 100 steps
python sample.py seed=1 config_file=./configs/mnist_discrete.yaml load_model= samples_shape="[4, 28, 28, 1]" n_steps=100 save_file=./samples_mnist.pt
```

The samples are stored as PyTorch tensors in the `save_file`, and can be visualized by loading them and then using the utilities `batch_to_images` and `batch_to_str` in `data.py`.
For example: 
```shell
# batch_to_images returns a matplotlib Figure object
python -c "import torch; from data import batch_to_images; batch_to_images(torch.load('./samples_mnist.pt')).savefig('mnist.png')"
```

## Visualizing Flow




## References

- Graves, Alex, et al. "Bayesian flow networks." arXiv preprint arXiv:2308.07037 (2023).
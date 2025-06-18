# Conditional Bayesian Flow Networks

This repository is based on the official code release for [Bayesian Flow Networks](https://arxiv.org/abs/2308.07037) by Alex Graves, Rupesh Kumar Srivastava, Timothy Atkinson and Faustino Gomez. We made some improvements and adjustments to enable the conditional generation.

## Reading Guide

- `checkpoints/` contains the compressed files of our experimental results of the conditional BFN, including models and training dynamics. Before using, do decompress the files to get our trained models.
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

Use `accelerate` to start training. Configs are defined in `configs/mnist_discrete_.yaml`.

```shell
# mnist experiment on 1 GPU
accelerate launch train.py
```
During training process the checkpoints are saved in `./checkpoints/BFN/`, including the latest (`last`) and the best (`best`) model.

## Testing
> [!NOTE]
> Depending on your GPU, you may wish to adjust the batch size used for testing in `test.py`.

After training is complete, the performance of the model can be evaluated on the test set using `test.py`. You need to specify the configuration file, the path to the model to load, the number of evaluation steps, and the number of repetitions.
```shell
# Compute 784-step loss on MNIST
python test.py seed=1 config_file=./configs/mnist_discrete_.yaml load_model=./checkpoints/BFN/last/pytorch_model.bin n_steps=784 n_repeats=20
```
> [!IMPORTANT]
> All computed results will be in nats-per-data-dimension. To convert to bits, divide by $\ln(2)$.

## Conditional Sampling

Use `sample.py` to generate new image samples according to the specified label:

```shell
# Sample 4 binarized MNIST images with label 1 using 100 steps
python sample.py seed=1 config_file=./configs/mnist_discrete_.yaml load_model=./checkpoints/BFN/last/pytorch_model.bin samples_shape="[4, 28, 28, 1]" label=1 n_steps=100 save_file=./samples_mnist_label_1.pt
```
* `label`: specifies the label of the number to be generated (for example, `label=1` means generating the number "1").

## Visualizing the generated samples
The samples are stored as PyTorch tensors in the `save_file`, and can be visualized by loading them and then using the utilities `batch_to_images` and `batch_to_str` in `data.py`.
For example: 
```shell
# batch_to_images returns a matplotlib Figure object
python -c "import torch; from data import batch_to_images; batch_to_images(torch.load('./samples_mnist_label_1.pt')).savefig('samples_mnist_label_1.png')"
```
* Make sure the path to the `.pt` file and the output `.png` file name match what you used in the previous step.

## Visualizing Flow
Use `visualize_flow.py` to analyze how the input and output distributions within the model evolve over time:
```shell
python visualize_flow.py
```
* Before running, make sure the `CONFIG_PATH`, `MODEL_CHECKPOINT_PATH`, and `IMAGE_INDICES_TO_VISUALIZE` variables in the script are set to your needs.


## References

- Graves, Alex, et al. "Bayesian flow networks." arXiv preprint arXiv:2308.07037 (2023).

<div align="center">

# `deepHSI`: A Deep Learning Toolbox for Hyperspectral Imaging

<!-- Python Version Badge -->

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![PyTorch Lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://www.pytorchlightning.ai/)

</div>

## **Overview**

`deepHSI` is a comprehensive toolbox for applying deep learning techniques to hyperspectral image analysis and processing. It is built on top of [PyTorch Lightning](https://lightning.ai/) and [Pyro](https://pyro.ai/), providing a scalable and flexible framework that facilitates the development and experimentation of models for those working with hyperspectral datasets. The design of `deepHSI` emphasizes modularity and ease of use, enabling researchers and practitioners to efficiently implement, train, and evaluate a wide range of deep learning architectures tailored to their specific hyperspectral imaging tasks.

<!-- ## **Features**

- **Modular Architecture**: Easily integrate new models and datasets.
- **Flexible Configuration**: Utilize Hydra configurations for flexible experiment setups.
- **Multi-GPU Support**: Scale your training to multi-GPU settings for faster processing.
- **Pre-built Modules**: Access a collection of pre-built modules tailored for HSI tasks.
- **Extensive Documentation**: Benefit from detailed documentation and examples. -->

## **Installation**

### Prerequisites

- Python 3.12
- Pip or Conda

### Steps

Clone the repository and set up the environment:

```shell
git clone git@github.com:skhan61/deepHSI.git
cd deepHSI

# Using pip
pip install -r requirements.txt

# Using Conda
conda env create -f environment.yml
conda activate deepHSI
```

## Complete Workflow for `deepHSI`

The `deepHSI` toolbox streamlines the process of using deep learning for hyperspectral imaging. The workflow is divided into key stages, each with dedicated components and functionalities.

### 1. Data Handling

Begin by organizing your hyperspectral imaging data. `deepHSI` offers a robust framework to seamlessly incorporate both custom and public datasets.

- **Built-in HSI Dataset Support**: `deepHSI` offers pre-configured support for a wide array of public hyperspectral imaging datasets in both remote sensing and medical imaging domains. This feature allows you to quickly engage in experiments and assessments with your models on established HSI datasets, facilitating immediate research and development progress.

- **Custom Dataset Class**: Design your dataset as a subclass of [PyTorch's Dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html). For guidance, refer to the structure outlined in [deepHSI.datamodule.HyperspectralDataset](deepHSI/datamodule/hyperspectral_datamodule.py).

- **Data Module** : Utilize [PyTorch Lightning's DataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) to structure your training, validation, and testing data splits efficiently. A template can be found in [deepHSI.datamodule.BaseHyperspectralDataModule](deepHSI/datamodule/hyperspectral_datamodule.py), which serves as a practical example for organizing hyperspectral data.

Through these steps, `deepHSI` ensures a smooth integration and handling of your hyperspectral imaging datasets, allowing for a streamlined workflow from data preprocessing to model training.

### 2. Model Development

`deepHSI` is equipped with a comprehensive suite for developing advanced models tailored to hyperspectral data analysis. It offers a variety of SOTA architectures for HSI and task-specific modules to cater to both standard and complex HSI processing tasks.

- **architectures**: This repository encompasses leading-edge architectures for Hyperspectral Imaging (HSI), providing a robust foundation for researchers to test and innovate with their datasets using the most advanced models in the field of HSI.

- **task_algos**: In `task_algos,` we have pre-implemented various tasks such as classification and Variational AutoEncoders (VAEs) using the PyTorch Lightning framework. For VAE development, we also integrate `Pyro`.

# **Tutorial**

For a hands-on introduction and detailed guidance on using our architectures and models, please refer to our tutorial notebook:

- [Tutorial 01: Getting Started](tutorials/notebooks/Tutorial-01.ipynb)
- [Tutorial 02: VAE](tutorials/notebooks/Tutorial-02.ipynb)

## **Contributing**

Contributions are welcome! Please refer to the [Contributing Guidelines](CONTRIBUTING.md) for more details.

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## **Acknowledgments**

This project is inspired by the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template). We express our gratitude to the authors for their foundational work.

## **Citation**

If `deepHSI` aids in your research, please cite it as follows:

```
@misc{deepHSI2024,
  author = {Sayem Khan},
  title = {deepHSI: A Deep Learning Toolbox for Hyperspectral Imaging},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/skhan61/deepHSI}}
}
```

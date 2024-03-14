<div align="center">

# `deepHSI`: A Deep Learning Toolbox for Hyperspectral Imaging

<!-- Python Version Badge -->
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)

<!-- PyTorch Version Badge -->
[![PyTorch Version](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)

<!-- PyTorch Lightning Version Badge -->
[![PyTorch Lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://www.pytorchlightning.ai/)

</div>


## **Overview**
`deepHSI` is an advanced toolbox designed for leveraging deep learning in the analysis and processing of hyperspectral images. Developed with PyTorch Lightning and configured through Hydra, it offers a scalable and modular framework for researchers and practitioners working with hyperspectral data.

## **Features**
- **Modular Architecture**: Easily integrate new models and datasets.
- **Flexible Configuration**: Utilize Hydra configurations for flexible experiment setups.
- **Multi-GPU Support**: Scale your training to multi-GPU settings for faster processing.
- **Pre-built Modules**: Access a collection of pre-built modules tailored for HSI tasks.
- **Extensive Documentation**: Benefit from detailed documentation and examples.

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

## **Getting Started**

### Training a Model
To train a model with the default configuration, execute:

```shell
python src/train.py
```

### Custom Configuration
To customize the training parameters or use an alternative configuration:

```shell
python src/train.py model=my_custom_model data=my_custom_dataset
```

### Evaluation
Evaluate a model on the test dataset:

```shell
python src/evaluate.py model=my_custom_model checkpoint=path/to/model.ckpt
```

## **Advanced Usage**

### Hyperparameter Tuning
Conduct hyperparameter tuning using Hydra's multi-run capability:

```shell
python src/train.py -m hparams_search=my_hyperparameter_search.yaml
```

### Distributed Training
Train on multiple GPUs using Distributed Data Parallel (DDP):

```shell
python src/train.py trainer=ddp trainer.gpus=4
```

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

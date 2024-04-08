
# Experiment Folder Documentation

Welcome to the **Experiment** folder, the cornerstone of our project where innovative machine learning experiments come to life. This directory is meticulously structured to facilitate the execution of diverse experiments, allowing users to test state-of-the-art (SOTA) models across various datasets.

## Directory Structure

Below is the structure of the `experiment` directory, providing a clear view of its organization:

```plaintext
experiments/
└── experiment_1/
    ├── config/
    │   ├── model/
    │   │   ├── model_a.yaml
    │   │   └── model_b.yaml
    │   └── config.yaml
    └── experiment.py
```
### Detailed Breakdown

#### Experiment 1

`experiment_1` is a specialized folder designed to offer users the ability to evaluate SOTA models on different datasets. The contents are as follows:

- **config/**: This directory houses all Hydra configuration files necessary for the experiment's execution.

    - **model/**: Contains YAML configuration files for each model. These files define model-specific parameters and settings.
        - `model_a.yaml`: Configuration for Model A.
        - `model_b.yaml`: Configuration for Model B.
    - **config.yaml**: The main configuration file that Hydra uses to orchestrate the experiment. It integrates model configurations and may include other settings such as data processing and evaluation criteria.

- **experiment.py**: The Python script that executes the experiment. It utilizes Hydra to parse configuration files and run the model on the selected dataset.

## Getting Started

To run an experiment, navigate to the respective experiment folder and execute the `experiment.py` script, ensuring Hydra is properly configured to read the necessary YAML files. For example:

```bash
cd experiment/experiment_1
python experiment.py

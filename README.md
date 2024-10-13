# Rapid Estimation of Left Ventricular Contractility with a Physics-informed Neural Network Inverse Modeling Approach #

This repository contains the implementation of a Physics-informed Neural Network (PINN) framework for the rapid prediction of hemodynamics in the closed-loop cardiovascular system. This model is described by a set of ODEs associated with a lumped parameter description of the circulatory system. The paper associated with this work is:
```
@article{NAGHAVI2024102995,
title = {Rapid estimation of left ventricular contractility with a physics-informed neural network inverse modeling approach},
journal = {Artificial Intelligence in Medicine},
pages = {102995},
year = {2024},
issn = {0933-3657},
doi = {https://doi.org/10.1016/j.artmed.2024.102995},
url = {https://www.sciencedirect.com/science/article/pii/S0933365724002379},
author = {Ehsan Naghavi and Haifeng Wang and Lei Fan and Jenny S. Choy and Ghassan Kassab and Seungik Baek and Lik-Chuan Lee},
}
```

## Installation ##
The code has been developed on Python 3.9. There may be compatibility issues with `torchrun` on Python 3.12 ([see this thread](https://github.com/pytorch/pytorch/issues/125990)).
We recommend creating a seperate Anaconda environment for the code ([getting started with Anaconda](https://docs.anaconda.com/free/anaconda/getting-started/)) and using the editable installation mode. Follow these steps to install:
```
git clone https://github.com/ehsanngh/lpm_pinn.git
cd lpm_pinn
pip install -e .
```

To verify if the code has been installed correctly, run the following command:
```
python tests/test_demo.py
```

## Training ##
The model can be trained on multiple GPUs in parallel using the following command:
```
torchrun --standalone --nproc_per_node=gpu ./plpm/scripts_internal/main_training.py [CONFIG_FILE] [NUM_ITERATIONS] [SAVE_INTERVAL] --batch_size [BATCH_SIZE]
```
where [CONFIG_FILE] primarily determines which physiological model parameters are defined as model inputs. For example, the model used in the paper with 10 input parameters was trained by the following command:
```
torchrun --standalone --nproc_per_node=gpu ./plpm/scripts_internal/main_training.py ./plpm/configs/10dmodel_config.json 5000 10 --batch_size 120
```
The trained models for four different configurations are saved in this repository. Additional models can be trained after their appropriate decoder is defined in the `ModelProcessing` class located at `plpm/classes/processing`.

## Using the trained models ##
The codes for utilizing the trained models are located in the folder `scripts`.

The `InvModelingSingleCASE.ipynb` notebook, located in the `scripts/InverseModeling` folder, can be used to perform inverse modeling on a single case. The waveforms data must be scaled between $0$ and $800$ $ms$. The data should be in the form of a `torch.tensor` with the shape `(num_timepoints, 3)`, where the columns represent time (ms), left ventricle volume (ml), and pressure (mmHg), respectively.

To perform inverse modeling on a group of data within a folder, use the `InvModelingCohorts.py` script located in the same folder.

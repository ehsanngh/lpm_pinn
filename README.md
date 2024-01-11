# Rapid Estimation of Left Ventricular Contractility with a Physics-Informed Neural Network (PINN) Inverse Modeling Approach #
This repository contains the implementation of a closed-loop cardiovascular circulation model. The model uses Physics-informed Neural Networks (PINNs) for efficient estimation of cardiac functions.

## Installation ##
We recommend using the editable installation mode. Follow these steps to install:
```
git clone https://github.com/ehsanngh/lpm_pinn.git
cd lpm_pinn
pip install -e .
pip install -r requirements.txt
```

To verify if the code has been installed correctly, run the following command:
```
python tests/test_demo.py
```

The codes for utilizing the trained model are located in the folder `scripts`.

## Inverse Modeling ##
The `InvModelingSingleCASE.ipynb` notebook, located in the `scripts/InverseModeling` folder, can be used to perform inverse modeling on a single case. The waveforms data must be scaled between $0$ and $800$ $ms$. The data should be in the form of a `torch.tensor` with the shape `(num_timepoints, 3)`, where the columns represent time (ms), left ventricle volume (ml), and pressure (mmHg), respectively.

To perform inverse modeling on a group of data within a folder, use the `InvModelingCohorts.py` script in the same folder.

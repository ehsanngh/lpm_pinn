import torch
from plpm import NumericalSolver, ModelProcessing, Config
from plpm import ModelWrapper
from pathlib import Path
current_path = Path(__file__).parent
config_file = current_path / '..' / 'plpm/configs/10dmodel_config.json'
config_file = config_file.resolve()
config = Config(config_file)
mp = ModelProcessing(config.parameters)

main_model = ModelWrapper(config=config, model_processor=mp)

# Defining the case
# Estimating with the trained pinn, if the dimension of the model is not 10,
# CASE should be modified accordingly.
mR_av, mR_ao, mC_ao, mR_art, mC_art = torch.tensor([1., 1., 1., 1., 1.])  # Multipliers
mR_vc, mC_vc, mR_mv, mEes_lv, mtrans_lv = torch.tensor([1., 1., 1., 1., 1.])

CASE = torch.tensor([[mR_av, mR_ao, mC_ao, mR_art, mC_art,
                      mR_vc, mC_vc, mR_mv, mEes_lv, mtrans_lv]])

# Calculating the numerical solution
numerical_solver = NumericalSolver(mp)
V_num = numerical_solver(CASE, dt=0.5, n_cycle=30)

V_pinn = main_model.predict_CASE(CASE, num_timepoints=1601)

error = torch.mean(torch.abs(V_pinn[:, 0] - V_num[:, 0]) / V_num[:, 0])
print('The RMAE between Numerical Method and the PINN for the left ventricle volume waveform is', error)

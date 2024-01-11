import torch
from plpm import numerical_solver, calculate_PLV
from plpm import ModelWrapper

space_dim = 10
model_file = 'MainModel10D'

main_model = ModelWrapper.Model(model_file=model_file, d=space_dim)
# Defining the case
mR_av, mR_ao, mC_ao, mR_art, mC_art = torch.tensor([1., 1., 1., 1., 1.])  # Multipliers
mR_vc, mC_vc, mR_mv, mEes_lv, mtrans_lv = torch.tensor([1., 1., 1., 1., 1.])

CASE = torch.tensor([[mR_av, mR_ao, mC_ao, mR_art, mC_art,
                      mR_vc, mC_vc, mR_mv, mEes_lv, mtrans_lv]])

# Calculating the numerical solution
V_num = numerical_solver(CASE, dt=0.5, n_cycle=30)

# Estimating with the trained pinn, if the dimension of the model is not 10,
# CASE should be modified accordingly.
V_pinn = main_model.predict_CASE(CASE, num_timepoints=801)

error = torch.mean(torch.abs(V_pinn[:, 0] - V_num[:, 0]) / V_num[:, 0])
print('The RMAE between Numerical Method and the PINN for the left ventricle volume waveform is ', error)

import torch
import numpy as np
from plpm import calculate_PLV, ModelWrapper, InverseModelingExpData
from plpm.constants.BaselineCase import base
import time


space_dim = 10
model_file = 'MainModel10DFinal'

main_model = ModelWrapper.Model(model_file=model_file, d=space_dim)
Scaling = main_model.Scaling

from plpm.classes.CollocationPoints import PINN_CP

# ----------------------------- Parameters -----------------------------------
dt = 0.5
num_timepoints = int(base.Tc//dt + 1)
num_cases = 100
seed = 12
# ----------------------------------------------------------------------------
pinn_cp = PINN_CP(num_timepoints=num_timepoints,
                  space_dimension=space_dim,
                  num_cases=num_cases)

sCASEs = pinn_cp._generate_sCASE(seed=seed)
sCASEs = torch.from_numpy(sCASEs)
CASEs = Scaling.InvCaseScaler(sCASEs)
t = torch.linspace(0., base.Tc, 801).view(-1, 1)

total_error = torch.zeros(CASEs.shape)
i = 0

for CASE in CASEs:
    CASE = CASE.view(1, -1)
    V_pred = main_model.predict_CASE(CASE, num_timepoints=len(t))
    P_LV = calculate_PLV(V_pred[:, 0], CASE[:, -2], CASE[:, -1]).view(-1, 1)
    data = torch.concat((t, V_pred[:, 0:1], P_LV), axis=1)
    inverse_modeling = InverseModelingExpData(model=main_model,
                                              data=data.detach().numpy(),
                                              verbose=False)
    bounds = [(0.3, 3.), (0.3, 3.), (0.3, 3.), (0.3, 3.), (0.3, 3.),
              (0.3, 3.), (0.3, 3.), (0.3, 3.), (0.3, 3.), (0.7, 1.35)]
    start = time.time()
    CASE_Estimated = inverse_modeling(bounds, seed=45, popsize=15, maxiter=500,
                                      strategy='best1bin', disp=False)
    total_error[i] = torch.abs(CASE - CASE_Estimated) / CASE
    if inverse_modeling.result.fun >= 1e-6:
        print(f"CASE {i} did NOT converge!")
    i += 1
    if i % 10 == 0:
        print(f"{i} CASEs passed.")
        np.save('data/InvUniqueness/totalerror.npy', total_error)

np.save('data/InvUniqueness/totalerror.npy', total_error)
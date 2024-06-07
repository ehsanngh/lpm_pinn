import torch
import numpy as np
from plpm import ModelProcessing, ModelWrapper, InverseModeling, Config
import time

from plpm.classes.CollocationPoints import PINN_CP

from pathlib import Path
current_path = Path(__file__).parent
config_file = current_path / '..' / '..' / 'plpm/configs/10dmodel_config.json'
config_file = config_file.resolve()
config = Config(config_file)
mp = ModelProcessing(config.parameters)

main_model = ModelWrapper(config=config, model_processor=mp)

# ----------------------------- Parameters -----------------------------------
dt = 0.5
Tc = mp.pmtrs['Tc']['base']
num_timepoints = int(Tc//dt + 1)
num_cases = 100
seed = 19
# ----------------------------------------------------------------------------
pinn_cp = PINN_CP(num_timepoints=num_timepoints,
                  space_dimension=mp.d,
                  num_cases=num_cases)

sCASEs = pinn_cp._generate_sCASE(seed=seed)
sCASEs = torch.from_numpy(sCASEs)
CASEs = mp.InvCaseScaler(sCASEs)
t = torch.linspace(0., Tc, 801).view(-1, 1)

total_error = torch.zeros(CASEs.shape)
i = 0

for CASE in CASEs:
    CASE = CASE.view(1, -1)
    V_pred = main_model.predict_CASE(CASE, num_timepoints=len(t))
    noise = 0.01 * V_pred.std(dim=0, keepdim=True) * torch.randn(V_pred.size())
    V_pred += noise
    P_LV = mp.calculate_PLV(V_pred[:, 0], CASE).view(-1, 1)
    noise = 0.01 * P_LV.std(dim=0, keepdim=True) * torch.randn(P_LV.size())
    P_LV += noise
    data = torch.concat((t, V_pred[:, 0:1], P_LV), axis=1)
    inverse_modeling = InverseModeling(model_wrapper=main_model,
                                       model_processor=mp,
                                       data=data.detach().numpy(),
                                       verbose=False)
    bounds = [(0.3, 3.), (0.3, 3.), (0.3, 3.), (0.3, 3.), (0.3, 3.),
              (0.3, 3.), (0.3, 3.), (0.3, 3.), (0.3, 3.), (0.7, 1.35)]
    start = time.time()
    CASE_Estimated = inverse_modeling(bounds, seed=45, popsize=15, maxiter=500,
                                      strategy='best1bin', disp=False)
    total_error[i] = torch.abs(CASE - CASE_Estimated) / CASE
    if inverse_modeling.result.fun >= 5e-3:
        print(f"CASE {i} did NOT converge! error: {inverse_modeling.result.fun}")
    i += 1
    if i % 10 == 0:
        print(f"{i} CASEs passed.")
        np.save('data/InvUniqueness/totalerror_withnoise01.npy', total_error)

np.save('data/InvUniqueness/totalerror_withnoise01.npy', total_error)
import torch
import numpy as np

from plpm import PINN_CP, ModelProcessing, NumericalSolver, Config

from pathlib import Path
current_path = Path(__file__).parent
config_file = current_path / '..' / 'plpm/configs/10dmodel_config.json'
config_file = config_file.resolve()
config = Config(config_file)
mp = ModelProcessing(config.parameters)
torch.set_default_dtype(torch.float64)

# ----------------------------- Parameters -----------------------------------
dt = 0.5
num_timepoints = int(mp.pmtrs['Tc']['base']//dt + 1)
num_cases = 1000
seed = 20
# ----------------------------------------------------------------------------
pinn_cp = PINN_CP(num_timepoints=num_timepoints,
                  space_dimension=mp.d,
                  num_cases=num_cases)

sCASEs = pinn_cp._generate_sCASE(seed=seed)
sCASEs = torch.from_numpy(sCASEs)
CASEs = mp.InvCaseScaler(sCASEs)

numerical_solver = NumericalSolver(model_processor=mp)

V = torch.zeros((CASEs.shape[0], num_timepoints, 5))
for i, CASE in enumerate(CASEs):
    CASE = torch.tensor([[CASE[0], CASE[1], CASE[2], CASE[3], CASE[4],
                          CASE[5], CASE[6], CASE[7], CASE[8], CASE[9]]])

    V[i] = numerical_solver(CASE, dt=dt, n_cycle=50)
    print(f'--------------------- CASE {i + 1} done ------------------------')

V = V.view(num_timepoints * num_cases, 5)
sV = mp.VolScaler(V)
sV = sV.view(num_cases, num_timepoints, 4)

nt = torch.linspace(0., 1., int(mp.pmtrs['Tc']['base']/dt) + 1)
sCASEs = sCASEs.numpy()
sV = sV.numpy()

np.savez("./data/numerical_data/ND_10D.npz",
         nt=nt,
         sCASEs=sCASEs,
         sV=sV)



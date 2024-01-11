import torch
import numpy as np

from plpm import PINN_CP, scaling, numerical_solver
from plpm.constants.BaselineCase import base
torch.set_default_dtype(torch.float64)

# ----------------------------- Parameters -----------------------------------
dt = 0.5
num_timepoints = int(base.Tc//dt + 1)
num_cases = 1000
space_dim = 10
seed = 20
# ----------------------------------------------------------------------------
Scaling = scaling(d=space_dim, sv_range=torch.tensor([-1., 1.]))
pinn_cp = PINN_CP(num_timepoints=num_timepoints,
                  space_dimension=space_dim,
                  num_cases=num_cases)

sCASEs = pinn_cp._generate_sCASE(seed=seed)
sCASEs = torch.from_numpy(sCASEs)
CASEs = Scaling.InvCaseScaler(sCASEs)

V = torch.zeros((CASEs.shape[0], num_timepoints, 5))
for i, CASE in enumerate(CASEs):
    CASE = torch.tensor([[CASE[0], CASE[1], CASE[2], CASE[3], CASE[4],
                          CASE[5], CASE[6], CASE[7], CASE[8], CASE[9]]])

    V[i] = numerical_solver(CASE, dt = dt, n_cycle=50, verbose=False)
    print(f'--------------------- CASE {i + 1} done ------------------------')

V = V.view(num_timepoints * num_cases, 5)
sV = Scaling.VolScaler(V)
sV = sV.view(num_cases, num_timepoints, 4)

nt = torch.linspace(0., 1., int(base.Tc/dt) + 1)
sCASEs = sCASEs.numpy()
sV = sV.numpy()

np.savez("./data/numerical_data/ND_10D_updatedBV.npz",
         nt=nt,
         sCASEs=sCASEs,
         sV=sV)



# % --------------------------------------------------------------------- % #


import torch
import numpy as np
from plpm import ModelProcessing, ModelWrapper, InverseModeling, Config
from scipy.interpolate import interp1d
import time
import os
import pandas as pd
from pathlib import Path
current_path = Path(__file__).parent
config_file = current_path / '..' / '..' / 'plpm/configs/5dmodel_config.json'
config_file = config_file.resolve()
config = Config(config_file)
mp = ModelProcessing(config.parameters)

main_model = ModelWrapper(config=config, model_processor=mp)

# bounds = [(0.3, 3.), (0.3, 3.), (0.3, 3.), (0.3, 3.), (0.3, 3.),
#           (0.3, 3.), (0.3, 3.), (0.3, 3.), (0.3, 3.), (0.7, 1.35)]

bounds = [(0.3, 3.), (0.3, 3.), (0.3, 3.), (0.7, 1.35), (0.7, 1.35)]

def metric(estimated, true):
    ss_res = torch.sum(torch.square(estimated - true))
    true_mean = true.mean()
    ss_tot = torch.sum(torch.square(true - true_mean))
    return 1 - ss_res / ss_tot

# %%
folder_path = 'data/ExperimentalData/processed'

AllEstCASES = torch.zeros((1, mp.d))
AllMetrics = torch.zeros((1, 2))
NamesList = []
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        NamesList.append(filename)
        case_path = os.path.join(folder_path, filename)
        data = np.loadtxt(case_path, delimiter=",", skiprows=1)

        num_timepoints = data.shape[0]
        num_cols = data.shape[1]
        interpolated_data = np.zeros((2 * num_timepoints, num_cols))
        for col in range(num_cols):
            interp_func = interp1d(np.arange(num_timepoints),
                                   data[:, col], 
                                   kind='linear')
            interpolated_data[:, col] = interp_func(
                np.linspace(0, num_timepoints - 1, 2 * num_timepoints))
        
        inverse_modeling = InverseModeling(model_wrapper=main_model,
                                           model_processor=mp,
                                           data=interpolated_data[:, :3])

        start = time.time()
        CASE_Estimated = inverse_modeling(bounds,
                                          seed=45,
                                          popsize=50,
                                          strategy='best1bin',
                                          maxiter=100,
                                          disp=False)
        print(f"Elapsed time: {time.time() - start} seconds")
        CASE = CASE_Estimated
        V_pinn = main_model.predict_CASE(CASE_Estimated)
        plv_pinn = mp.calculate_PLV(V_pinn[:, 0], CASE)
        V_pinn_metric = main_model.predict_CASE(CASE_Estimated, nt=torch.from_numpy(data[:, 0:1]) / mp.Tc)
        plv_pinn_metric = mp.calculate_PLV(V_pinn_metric[:, 0], CASE)
        
        vlv_true = torch.from_numpy(data[:, 1])
        VLV_error = metric(V_pinn_metric[:, 0], vlv_true)
        
        plv_true = torch.from_numpy(data[:, 2])
        plv_error = metric(plv_pinn_metric, plv_true)
        metrics = torch.tensor([VLV_error, plv_error])

        AllEstCASES = torch.vstack((AllEstCASES, CASE_Estimated))
        AllMetrics = torch.vstack((AllMetrics, metrics))


InvResults = pd.DataFrame(index=NamesList,
                        #   columns=['$R_{av}$', '$R_{ao}$', '$C_{ao}$', '$R_{art}$', '$C_{art}$',
                        #            '$R_{vc}$', '$C_{vc}$', '$R_{mv}$', '$E_{es}$', '$t_{tr}$'],
                          columns=['$C_{art}$', '$C_{vc}$', '$E_{es}$', '$T_{max}$', '$t_{tr}$'],
                          data=AllEstCASES[1:, :].numpy())

InvResults.to_csv('data/InvModeling/InvResults5d.csv', index=True)

MtcResults = pd.DataFrame(index=NamesList,
                          columns=['$R^2_{Vlv}$', '$R^2_{Plv}$'],
                          data=AllMetrics[1:, :].numpy())

MtcResults.to_csv('data/InvModeling/MtcResults5d.csv', index=True)

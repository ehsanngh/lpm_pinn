# %%

import torch
from plpm import ModelWrapper, ModelProcessing, Config
import time
import numpy as np
from SALib import ProblemSpec
from pathlib import Path
current_path = Path(__file__).parent
config_file = current_path / '..' / '..' / 'plpm/configs/10dmodel_config.json'
config_file = config_file.resolve()
config = Config(config_file)
mp = ModelProcessing(config.parameters)

main_model = ModelWrapper(config=config, model_processor=mp)

sp = ProblemSpec({

    'names': ['nmR_av', 'nmR_ao', 'nmC_ao', 'nmR_art', 'nmC_art',
              'nmR_vc', 'nmC_vc', 'nmR_mv', 'nmEes_lv', 'nmt_tr'],
    'bounds': [[0, 1],
               [0, 1],
               [0, 1],
               [0, 1],
               [0, 1],
               [0, 1],
               [0, 1],
               [0, 1],
               [0, 1],
               [0, 1]]
    })

# %%
nt = torch.linspace(0.0, 1.0, 201)
def wrapped_model(parameters, nt=nt):
    y = np.zeros((parameters.shape[0], nt.shape[0]))
    nt = nt.reshape((-1, 1))
    for i, (nmR_av, nmR_ao, nmC_ao, nmR_art, nmC_art,
            nmR_vc, nmC_vc, nmR_mv, nmEes_lv, nmtrans_lv) in enumerate(parameters):
        sCASE = torch.tensor([[nmR_av, nmR_ao, nmC_ao, nmR_art, nmC_art,
                               nmR_vc, nmC_vc, nmR_mv, nmEes_lv, nmtrans_lv]])
        V_pinn = main_model.predict_sCASE(sCASE, nt)
        y[i,:] = V_pinn.numpy()[:, output_num]
    return y

S1 = np.zeros((5, 10))
ST = np.zeros((5, 10))
S2 = np.zeros((5, 10, 10))

# %%
start = time.time()
for output_num in range(5):
    (
        sp.sample_sobol(8192*2)
        .evaluate(wrapped_model)
        .analyze_sobol()
    )

    S1s = np.array([sp.analysis[_y]['S1'] for _y in sp['outputs']])
    STs = np.array([sp.analysis[_y]['ST'] for _y in sp['outputs']])
    S2s = np.array([sp.analysis[_y]['S2'] for _y in sp['outputs']])
    S1[output_num, :] = S1s.mean(axis=0)
    ST[output_num, :] = STs.mean(axis=0)
    S2[output_num, :, :] = S2s.mean(axis=0)
    print('Elapsed Time:', time.time() - start, 'seconds')
    start = time.time()

np.savez('data/SensitivityAnalysis/SA10D_Vs.npz', S1=S1, ST=ST, S2=S2)

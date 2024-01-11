import torch
from plpm.classes.NumericalMethod import calculate_PLV
from plpm.constants.BaselineCase import base
from scipy.optimize import differential_evolution

class InverseModelingExpData:
    def __init__(self, model, data, verbose=True):
        self.model = model
        self.space_dim = self.model.space_dim
        self.data = data
        self.num_signals = data.shape[1]
        self.nt = torch.from_numpy(data[:, 0:1]) / base.Tc
        self.vlv = torch.from_numpy(data[:, 1])
        self.loss1 = torch.tensor(0.)
        self.loss2 = torch.tensor(0.)
        self.loss3 = torch.tensor(0.)
        if verbose:
            print('V_LV loaded')
        if self.num_signals > 2:
            self.Plv = torch.from_numpy(data[:, 2])
            if verbose:
                print('P_LV loaded')
            if self.num_signals > 3:
                self.Pao = torch.from_numpy(data[:, 3])
                if verbose:
                    print('P_ao loaded')

    def metric(self, estimated, true):
        ss_res = torch.sum(torch.square(estimated - true))
        true_mean = true.mean()
        ss_tot = torch.sum(torch.square(true - true_mean))
        return ss_res / ss_tot

    def loss_wrapper(self, x):
        CASE_Estimated = torch.from_numpy(x).reshape((1, self.space_dim))
        if self.space_dim == 4:
            mC_ao = torch.tensor(1.)
            mEes_lv = CASE_Estimated[:, -1]
            mtrans_lv = torch.tensor(1.)
        
        elif self.space_dim == 6:
            mC_ao = torch.tensor(1.)
            mEes_lv = CASE_Estimated[:, -2]
            mtrans_lv = CASE_Estimated[:, -1]

        elif self.space_dim == 8:
            mC_ao = CASE_Estimated[:, 1]
            mEes_lv = CASE_Estimated[:, -2]
            mtrans_lv = CASE_Estimated[:, -1]
        
        elif self.space_dim == 10:
            mC_ao = CASE_Estimated[:, 2]
            mEes_lv = CASE_Estimated[:, -2]
            mtrans_lv = CASE_Estimated[:, -1]
            
        else:
            raise ValueError("Invalid value of d. \
                             Allowed values are 4, 6, 8, and 10.") 
        V_pinn = self.model.predict_CASE(CASE_Estimated, nt=self.nt)
        w1 = 1.
        self.loss1 = self.metric(V_pinn[:, 0], self.vlv)
    
        w2 = 0.
        w3 = 0.

        if self.num_signals > 2:
            pinnP_lv = calculate_PLV(V_pinn[:, 0], mEes_lv, mtrans_lv)
            w2 = 1.
            self.loss2 = self.metric(pinnP_lv, self.Plv)
            
            if self.num_signals > 3:
                C_ao = base.C_ao *  mC_ao
                pinnP_ao = (V_pinn[:, 1] - base.v_ao_r) / C_ao
                w3 = 1.
                self.loss3 = self.metric(pinnP_ao * 0.0075, self.Pao)

        loss = (w1 * self.loss1 + w2 * self.loss2 + w3 * self.loss3
                ) / (w1 + w2 + w3)
        return loss.item()
    
    def __call__(self, bounds, seed, popsize, strategy, maxiter, disp=True):
        result = differential_evolution(self.loss_wrapper,
                                        bounds,
                                        seed=seed,
                                        popsize=popsize,
                                        strategy=strategy,
                                        maxiter=maxiter,
                                        disp=disp)
        self.result = result
        CASE_Estimated = torch.from_numpy(result.x).reshape((1, self.space_dim))
        return CASE_Estimated
    
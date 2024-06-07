import torch
from plpm.classes import ModelProcessing
from plpm.models import ModelWrapper
from scipy.optimize import differential_evolution
import numpy as np

class InverseModeling:
    """
    This callable class performs inverse modeling on given volume and
    pressure waveforms using an evolutionary method.

    Inputs to initialize the class:
    - model_wrapper (ModelWrapper): This should be a trained model that
    is wrapped by the ModelWrapper class before being passed to this class.
    - model_processor (ModelProcessing): The model processor.
    - data (numpy.ndarray): A numpy array in the format of (num_timepoints, 2)
    to (num_timepoints, 4). The second, third, and fourth columns represent
    V_LV, P_LV, and P_ao, respectively. P_LV and P_ao are optional.

    Output:
    - CASE_Estimated: The estimated input model parameters that maximize the
    R^2 values between the outputs of the trained PINN model and the data
    """
    def __init__(self,
                 model_wrapper: ModelWrapper,
                 model_processor: ModelProcessing,
                 data: np.ndarray, verbose=True):
        self.model = model_wrapper
        self.mp = model_processor
        self.space_dim = self.mp.d
        self.data = data
        self.num_signals = data.shape[1]
        self.nt = torch.from_numpy(data[:, 0:1]) / self.mp.pmtrs['Tc']['base']
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
        """
        Given estimated and true signals, it calculates the R^2 value.
        """
        ss_res = torch.sum(torch.square(estimated - true))
        true_mean = true.mean()
        ss_tot = torch.sum(torch.square(true - true_mean))
        return ss_res / ss_tot

    def loss_wrapper(self, x):
        """
        This function wraps the loss for the inverse modeling
        optimization problem.

        Input:
        - x: The estimated case as a numpy array

        Output:
        - loss: The loss of the optimization problem, returned as a
        scalar numpy value
        """
        CASE_Estimated = torch.from_numpy(x).reshape((1, self.space_dim))
        V_pinn = self.model.predict_CASE(CASE_Estimated, nt=self.nt)
        w1 = 1.
        self.loss1 = self.metric(V_pinn[:, 0], self.vlv)
    
        w2 = 0.
        w3 = 0.

        if self.num_signals > 2:
            pinnP_lv = self.mp.calculate_PLV(V_pinn[:, 0], CASE_Estimated)
            w2 = 1.
            self.loss2 = self.metric(pinnP_lv, self.Plv)
            
            if self.num_signals > 3:
                pinnP_ao = (V_pinn[:, 1] - self.mp.v_ao_r) / self.mp.C_ao
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
    
import torch
from torch.utils.data import Dataset
from scipy.stats.qmc import LatinHypercube
import numpy as np
from pkg_resources import resource_stream

class Residual_DS(Dataset):
    """
    Returns the input x as a PyTorch dataset that can be
    conveniently distributed among multiple GPUs.

    Input:
    - sCASEs: A NumPy array of shape (num_cases, input_size),
    (num_cases * num_timepoints, input_size), or
    (num_cases, input_size, num_timepoints)

    Output:
    - A PyTorch dataset that can be distributed on multiple GPUs (DDP)
    """
    def __init__(self, x):
        self.size = len(x)
        self.x = [torch.from_numpy(x[i]) for i in range(self.size)]
        
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.x[index]
    

class NumericalData_DS(Dataset):
    """
    Returns a PyTorch dataset of the numerical data that can be
    conveniently distributed among multiple GPUs.

    Input:
    - x: A NumPy array of shape (num_cases, input_size),
    (num_cases * num_timepoints, input_size), or
    (:, input_size, num_timepoints)
    - y: A NumPy array of the scaled numerical results in the shape
    (num_cases, num_timpoints, 4)

    Output:
    - A PyTorch dataset that can be distributed on multiple GPUs
    """
    def __init__(self, x, y):
        self.size = len(x)
        self.x = [torch.from_numpy(x[i]) for i in range(self.size)]
        self.y = [torch.from_numpy(y[i]) for i in range(self.size)]
        
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    

# ----------------------------------------------------------------------------
class PINN_CP:
    def __init__(self,
                 num_timepoints=401,
                 space_dimension=10,
                 num_cases=20000):
        self.num_timepoints = num_timepoints
        self.num_cases = num_cases
        self.space_dimension = space_dimension
    
    def normalized_timepoints(self):
        nt = torch.linspace(
            0., 1., self.num_timepoints).reshape((-1, 1)).requires_grad_()
        return nt

    def _generate_sCASE(self, seed):   
        LHC_Sampler = LatinHypercube(d=self.space_dimension, seed=seed)
        sCASEs = LHC_Sampler.random(n=self.num_cases)  # Scaled Cases
        return sCASEs  # np array
    
    def build_sCASEsDS(self, seed=0):
        return Residual_DS(self._generate_sCASE(seed=seed))  # Tensor Dataset

    def build_NumericalDataDS(self,
                              data_file='numerical_data1000',
                              step=1):
        numerical_data = np.load(
        resource_stream('plpm', '../data/numerical_data/' + data_file +'.npz'))
        sCASEs = numerical_data['sCASEs']
        sV = numerical_data['sV'][:, ::step, :]
        nt_numerical = np.linspace(0., 1., num=sV.shape[1]).reshape((-1, 1))
        return torch.from_numpy(nt_numerical), NumericalData_DS(sCASEs, sV)


# ----------------------------------------------------------------------------
class FNO_CP:
    '''
    For replacing FCNN with FNO (under development)
    '''
    def __init__(self,
                 num_timepoints=401,
                 space_dimension=10,
                 num_cases=20000):
        self.num_timepoints = num_timepoints
        self.num_cases = num_cases
        self.space_dimension = space_dimension

    def normalized_timepoints(self):
        nt = np.linspace(
            0., 1., self.num_timepoints)
        return nt
    
    def _generate_sCASE(self, seed):   
        LHC_Sampler = LatinHypercube(d=self.space_dimension, seed=seed)
        sCASEs = LHC_Sampler.random(n=self.num_cases)  # Scaled Cases
        return sCASEs

    def build_XTrain(self, sCASEs, nt):
        X_Train = np.zeros((self.num_cases, 11, self.num_timepoints))
        X_Train[:, 10, :] = nt
        for i, sCASE in enumerate(sCASEs):
            X_Train[i, :10, :] = np.tile(sCASE[:, np.newaxis], (1, self.num_timepoints))
        return X_Train
    
    def build_XTrainDS(self, seed=0):
        sCASEs = self._generate_sCASE(seed)
        nt = self.normalized_timepoints()
        X_Train = self.build_XTrain(sCASEs, nt)
        return Residual_DS(X_Train)

    def build_NumericalDataDS(self,
                              data_file='numerical_data1000',
                              step=1):
        numerical_data = np.load(
        resource_stream('plpm', '../data/numerical_data/' + data_file +'.npz'))
        nt_numerical = numerical_data['nt'][::step]
        sCASEs_numerical = numerical_data['sCASEs']
        sV = numerical_data['sV'][:, ::step, :]
        X_Train = self.build_XTrain(sCASEs_numerical, nt_numerical)
        return NumericalData_DS(X_Train, sV)




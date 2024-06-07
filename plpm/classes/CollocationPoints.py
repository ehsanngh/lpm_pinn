import torch
from torch.utils.data import Dataset
from scipy.stats.qmc import LatinHypercube
import numpy as np
from pathlib import Path
current_path = Path(__file__).parent

class Residual_DS(Dataset):
    """
    This class converts the input 'x' into a PyTorch dataset that can be
    conveniently distributed among multiple GPUs.

    Input:
    - sCASEs (numpy.ndarray): A NumPy array with one of the following shapes:
        - (num_cases, input_size)
        - (num_cases * num_timepoints, input_size)
        - (num_cases, input_size, num_timepoints)

    Returns:
    - PyTorch Dataset: A dataset that can be distributed on multiple GPUs
    using Distributed Data Parallel (DDP)
    """
    def __init__(self, sCASEs):
        self.size = len(sCASEs)
        self.x = [torch.from_numpy(sCASEs[i]) for i in range(self.size)]
        
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.x[index]
    

class NumericalData_DS(Dataset):
    """
    This class converts the numerical data into a PyTorch dataset that can
    be conveniently distributed among multiple GPUs.

    Parameters:
    - x (numpy.ndarray): A NumPy array with one of the following shapes:
        - (num_cases, input_size)
        - (num_cases * num_timepoints, input_size)
        - (num_cases, input_size, num_timepoints)
    - y (numpy.ndarray): A NumPy array of the scaled numerical results with
    shape (num_cases, num_timpoints, 4)

    Returns:
    - PyTorch Dataset: A dataset that can be distributed on multiple GPUs (DDP)
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
    """
    This class contains methods for preparing training data points.

    Inputs:
    - num_timepoints (int): Determines the number of timepoints
    - space_dimension (int): Determines the number of model inputs
    - num_cases (int): Determines the number of cases to sample the
    parameter space using the Latin Hypercube method
    """
    def __init__(self,
                 num_timepoints=401,
                 space_dimension=10,
                 num_cases=10000):
        self.num_tp = num_timepoints
        self.num_cases = num_cases
        self.space_dimension = space_dimension
    
    def normalized_timepoints(self):
        nt = torch.linspace(0., 1., self.num_tp).view((-1, 1)).requires_grad_()
        return nt

    def _generate_sCASE(self, seed):
        """
        Given a seed as input, this function returns the sample points of
        the parameter space, which has the dimension of 'self.space_dimension'. 

        Input:
        - seed: The seed for the random number generator

        Returns:
        - numpy.ndarray: A NumPy array of scaled cases (ranging from 0 to 1)
        with shape (self.num_cases, self.space_dimension)
        """
  
        LHC_Sampler = LatinHypercube(d=self.space_dimension, seed=seed)
        sCASEs = LHC_Sampler.random(n=self.num_cases)
        return sCASEs
    
    def build_sCASEsDS(self, seed=0):
        """
        Returns a PyTorch Dataset of scaled cases that can be distributed
        on multiple GPUs
        """
        return Residual_DS(self._generate_sCASE(seed=seed))

    def build_NumericalDataDS(self, data_file, step=4):
        """
        This function returns a tensor dataset of 'numerical_data' that can
        be conveniently distributed among multiple GPUs.

        Inputs:
        - data_file (str): This can either be the absolute directory of
        the 'numerical_data', which must end with '.npz', or just the
        name of the 'numerical_data' file if it is already located in
        the 'data/numerical_data' folder. For example, 'ND_10D_updatedBV'.
        - step (int, optional): Determines how many time points are to
        be included in the training data. By default, every 4 timepoints
        are included.

        Returns:
        - PyTorch Dataset: A dataset that can be distributed on multiple GPUs
        """

        if data_file[-4:] != '.npz':
            data_file = current_path / '..' / '..' / 'data' / \
                'numerical_data' / (data_file + '.npz')
            data_file = data_file.resolve()
        numerical_data = np.load(data_file)
        sCASEs = numerical_data['sCASEs']
        sV = numerical_data['sV'][:, ::step, :]
        nt_numerical = np.linspace(0., 1., num=sV.shape[1]).reshape((-1, 1))
        return torch.from_numpy(nt_numerical), NumericalData_DS(sCASEs, sV)

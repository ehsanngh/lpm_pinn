import torch
import numpy as np
from torch import nn
import torch.nn.init as init


def fourier_transform(inputs):
    """First 12 basis functions from the Fourier series"""
    nt = inputs[:, 0:1]
    w = 2 * np.pi    
    return torch.concat(
        (
            torch.cos(w * nt),
            torch.sin(w * nt),
            torch.cos(2 * w * nt),
            torch.sin(2 * w * nt),
            torch.cos(3 * w * nt),
            torch.sin(3 * w * nt),
            torch.cos(4 * w * nt),
            torch.sin(4 * w * nt),
            torch.cos(5 * w * nt),
            torch.sin(5 * w * nt),
            torch.cos(6 * w * nt),
            torch.sin(6 * w * nt),
        ),
        axis=1,
    )


def create_network(input_size, num_layers, hidden_size, output_size,
                   activation_func):
    layers = []
    layers.extend([nn.Linear(input_size, hidden_size),
                    activation_func
                    ])

    for _ in range(num_layers - 1):
        layers.extend([nn.Linear(hidden_size, hidden_size),
                        activation_func
                        ])
    
    layers.append(nn.Linear(hidden_size, output_size))
    return nn.Sequential(*layers)


class BaselinePINN(nn.Module):
    '''
    The only input is time.
    '''
    def __init__(self,
                 num_layers=5,
                 hidden_size=16,
                 activation_func = nn.Tanh()):
        super(BaselinePINN, self).__init__()

        input_size = 12
        output_size = 1
        self.fourier_transform = fourier_transform
        self.create_network = create_network
        self.v1 = self.create_network(input_size, num_layers, hidden_size,
                                      output_size, activation_func)
        self.v2 = self.create_network(input_size, num_layers, hidden_size,
                                      output_size, activation_func)
        self.v3 = self.create_network(input_size, num_layers, hidden_size,
                                      output_size, activation_func)
        self.v4 = self.create_network(input_size, num_layers, hidden_size,
                                      output_size, activation_func)
        self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Apply different weight initialization for each network
                if m in self.v1.modules():
                    init.xavier_uniform_(m.weight, gain=1.)
                    init.constant_(m.bias, 0.)
                elif m in self.v2.modules():
                    init.xavier_normal_(m.weight, gain=1.)
                    init.zeros_(m.bias)
                elif m in self.v3.modules():
                    init.xavier_normal_(m.weight, gain=1.)
                    init.constant_(m.bias, 0.)
                elif m in self.v4.modules():
                    init.xavier_uniform_(m.weight, gain=1.)
                    init.constant_(m.bias, 0.)

    def forward(self, nt):
        nt_fourier = self.fourier_transform(nt)
        svlv = self.v1(nt_fourier)
        svao = self.v2(nt_fourier)
        svart = self.v3(nt_fourier)
        svla = self.v4(nt_fourier)
        pred = torch.hstack((svlv, svao, svart, svla))
        return pred
    

class MainPINN(nn.Module):
    def __init__(self,
                 input_size=10,
                 num_layers=5,
                 hidden_size=256,
                 activation_func = nn.Tanh()):
        super(MainPINN, self).__init__()

        input_size = 12 + input_size
        output_size = 1
        self.fourier_transform = fourier_transform
        self.create_network = create_network
        self.v1 = self.create_network(input_size, num_layers, hidden_size,
                                      output_size, activation_func)
        self.v2 = self.create_network(input_size, num_layers, hidden_size,
                                      output_size, activation_func)
        self.v3 = self.create_network(input_size, num_layers, hidden_size,
                                      output_size, activation_func)
        self.v4 = self.create_network(input_size, num_layers, hidden_size,
                                      output_size, activation_func)
        self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Apply different weight initialization for each network
                if m in self.v1.modules():
                    init.xavier_uniform_(m.weight, gain=1.)
                    init.constant_(m.bias, 0.)
                elif m in self.v2.modules():
                    init.xavier_normal_(m.weight, gain=1.)
                    init.zeros_(m.bias)
                elif m in self.v3.modules():
                    init.xavier_normal_(m.weight, gain=1.)
                    init.constant_(m.bias, 0.)
                elif m in self.v4.modules():
                    init.xavier_uniform_(m.weight, gain=1.)
                    init.constant_(m.bias, 0.)

    def forward(self, sCASEs, nt):
        nt_fourier = self.fourier_transform(nt)
        inputs = torch.concat((nt_fourier, sCASEs), axis=1)
        svlv = self.v1(inputs)
        svao = self.v2(inputs)
        svart = self.v3(inputs)
        svla = self.v4(inputs)
        pred = torch.hstack((svlv, svao, svart, svla))
        return pred
    

    

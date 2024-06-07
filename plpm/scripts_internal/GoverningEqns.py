import torch
from torch import nn
import numpy as np
from plpm import ModelProcessing


class NumericalDerivative(object):
    def __init__(self, L=1., fix_x_bnd=False):
        super().__init__()

        self.d = 1
        self.fix_x_bnd = fix_x_bnd

        if isinstance(L, float):
            self.L = [L]*self.d
        else:
            self.L = L

    def central_diff_1d(self, x, h):
        dx = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1))/(2.0*h)

        if self.fix_x_bnd:
            dx[...,0] = (x[...,1] - x[...,0])/h
            dx[...,-1] = (x[...,-1] - x[...,-2])/h
        
        return dx


    def uniform_h(self, x):
        h = [0.0]*1
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j]/x.size(-j)
        
        return h

    def __call__(self, x):
        h=self.uniform_h(x)
        x_x = self.central_diff_1d(x, h[0])
        return x_x

class GoverningEqns:
    def __init__(self,
                 model_processor: ModelProcessing,
                 num_derivative: bool=False):
        self.num_derivative = num_derivative
        self.mp = model_processor
        self.parameter_update = self.mp.parameter_update
            
    def __call__(self, sCASE, nt, sV, alpha = 0.01):
        CASE = self.mp.InvCaseScaler(sCASE)
        self.parameter_update(CASE)

        V = self.mp.InvVolScaler(sV, self.mp.V_total)
        t = nt * self.mp.Tc
        
        
        p_ao = (V[:, 1:2] - self.mp.v_ao_r) / self.mp.C_ao
        p_art = (V[:, 2:3] - self.mp.v_art_r) / self.mp.C_art
        p_vc = (V[:, 3:4] - self.mp.v_vc_r) / self.mp.C_vc
            
        Tmax, tau, trans = self.mp.Tmax_lv, self.mp.tau_lv, self.mp.trans_lv
        e_lv = torch.where(t <= trans,
                    0.5 * (1 - torch.cos(np.pi / Tmax * t)),
                    0.5 * (1 - torch.cos(np.pi / Tmax * trans)) * \
                        torch.exp((-t + trans) / tau))
        p_lv = e_lv * self.mp.Ees_lv * (V[:, :1] - self.mp.v_lv_r) + (1 - e_lv) \
            * self.mp.A_lv * (torch.exp(self.mp.B_lv * (V[:, :1] - self.mp.v_lv_r)) - 1)
        
        
        t_la = torch.where(t<700, t + 100, t + 100 - self.mp.Tc)
        Tmax, tau, trans = self.mp.Tmax_la, self.mp.tau_la, self.mp.trans_la
        e_la = torch.where(t_la <= trans,
                    0.5 * (1 - torch.cos(np.pi / Tmax * t_la)),
                    0.5 * (1 - torch.cos(np.pi / Tmax * trans)) * \
                        torch.exp((-t_la + trans) / tau))
        p_la = e_la * self.mp.Ees_la * (V[:, 4:] - self.mp.v_la_r) + (1 - e_la) \
            * self.mp.A_la * (torch.exp(self.mp.B_la * (V[:, 4:] - self.mp.v_la_r)) - 1)
        
        """ calculate the maximum of the pressure difference and zero using
        softplus with regularization """
        q_av = (1 / alpha) * nn.functional.softplus(alpha * (p_lv - p_ao)) / self.mp.R_av
        q_mv = (1 / alpha) * nn.functional.softplus(alpha * (p_la - p_lv)) / self.mp.R_mv
        
        q_ao = (p_ao - p_art) / self.mp.R_ao
        q_art = (p_art - p_vc) / self.mp.R_art
        q_vc = (p_vc - p_la) / self.mp.R_vc

        if self.num_derivative:
            dnvdnt = NumericalDerivative(L=1.)
            ddntsv_lv = dnvdnt(sV[:, 0:1])
            ddnt_sv_ao = dnvdnt(sV[:, 1:2])
            ddnt_sv_art = dnvdnt(sV[:, 2:3])
            ddnt_sv_la = dnvdnt(sV[:, 3:4])

        else:
            ddntsv_lv = torch.autograd.grad(
                sV[:, 0:1],
                nt,
                grad_outputs=torch.ones_like(sV[:, 0:1]),
                create_graph=True)[0][:, 0:1]
            
            ddnt_sv_ao = torch.autograd.grad(
                sV[:, 1:2],
                nt,
                grad_outputs=torch.ones_like(sV[:, 1:2]),
                create_graph=True)[0][:, 0:1]
            
            ddnt_sv_art = torch.autograd.grad(
                sV[:, 2:3],
                nt,
                grad_outputs=torch.ones_like(sV[:, 2:3]),
                create_graph=True)[0][:, 0:1]
            
            ddnt_sv_la = torch.autograd.grad(
                sV[:, 3:4],
                nt,
                grad_outputs=torch.ones_like(sV[:, 3:4]),
                create_graph=True)[0][:, 0:1]
        
        ddnt_sV = torch.cat((ddntsv_lv, ddnt_sv_ao, ddnt_sv_art, ddnt_sv_la), dim=1)
        ddt_V = (self.mp.Vs_max - self.mp.Vs_min) / (
            self.mp.sv_range[1] - self.mp.sv_range[0]) / self.mp.Tc * ddnt_sV
        ddt_v_vc = - ddt_V.sum(dim=1, keepdim=True)

        ODE1 = ddt_V[:, :1] - (q_mv - q_av)
        ODE2 = ddt_V[:, 1:2] - (q_av - q_ao)
        ODE3 = ddt_V[:, 2:3] - (q_ao - q_art)
        ODE4 = ddt_v_vc - (q_art - q_vc)
        ODE5 = ddt_V[:, 3:] -(q_vc - q_mv)

        return torch.cat((ODE1, ODE2, ODE3, ODE4, ODE5), axis=1)


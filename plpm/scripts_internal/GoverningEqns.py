import torch
from torch import nn
import numpy as np
from plpm.constants.BaselineCase import base
from plpm import scaling


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
                 d=10,
                 sv_range=torch.tensor([-1., 1.]),
                 num_derivative=False):
        self.d = d
        self.sv_range = sv_range
        self.Scaling = scaling(d=d, sv_range=sv_range)
        self.num_derivative = num_derivative

    def __call__(self, sCASE, nt, sV, alpha = 0.01):
        Scaling = self.Scaling
        if self.d == 0:
            mR_ao = torch.tensor(1.)
            mC_ao = torch.tensor(1.)
            mC_vc = torch.tensor(1.)
            mEes_lv = torch.tensor(1.)
            mR_av = torch.tensor(1.)
            mR_art = torch.tensor(1.)
            mC_art = torch.tensor(1.)
            mR_vc = torch.tensor(1.)
            mR_mv = torch.tensor(1.)
            mtrans_lv = torch.tensor(1.)

        else:
            CASE = Scaling.InvCaseScaler(sCASE)

            if self.d == 4:
                mR_art, mC_art, mC_vc, mEes_lv = torch.split(CASE, 1, dim=1)
                mR_av = torch.tensor(1.)
                mR_ao = torch.tensor(1.)
                mC_ao = torch.tensor(1.)
                mR_vc = torch.tensor(1.)
                mR_mv = torch.tensor(1.)
                mtrans_lv = torch.tensor(1.)

            elif self.d == 6:
                mR_art, mC_art, mC_vc, mR_mv, mEes_lv, \
                    mtrans_lv = torch.split(CASE, 1, dim=1)
                mR_av = torch.tensor(1.)
                mR_ao = torch.tensor(1.)
                mC_ao = torch.tensor(1.)
                mR_vc = torch.tensor(1.)

            elif self.d == 8:
                mR_ao, mC_ao, mR_art, mC_art, mC_vc, mR_mv, mEes_lv, \
                    mtrans_lv = torch.split(CASE, 1, dim=1)
                mR_av = torch.tensor(1.)
                mR_vc = torch.tensor(1.)

            elif self.d == 10:
                mR_av, mR_ao, mC_ao, mR_art, mC_art, mR_vc, mC_vc, mR_mv, \
                    mEes_lv, mtrans_lv = torch.split(CASE, 1, dim=1)

            else:
                raise ValueError("Invalid value of d. \
                                Allowed values are 1, 4, 6, 8, and 10.")
        
        V = Scaling.InvVolScaler(sV)
        v_lv_p, v_ao_p, v_art_p, v_vc_p, v_la_p = torch.split(V, 1, dim=1)
        
        R_av = base.R_av * mR_av
        
        R_ao = base.R_ao * mR_ao
        C_ao = base.C_ao * mC_ao

        R_art = base.R_art * mR_art
        C_art = base.C_art * mC_art

        R_vc = base.R_vc * mR_vc
        C_vc = base.C_vc * mC_vc

        R_mv = base.R_mv * mR_mv

        Ees_lv = base.Ees_lv * mEes_lv
        Ees_la = base.Ees_la
        trans_lv = mtrans_lv * base.trans_lv

        Tc = base.Tc
        t = nt * Tc
        
        p_ao = (v_ao_p - base.v_ao_r) / C_ao
        p_art = (v_art_p - base.v_art_r) / C_art
        p_vc = (v_vc_p - base.v_vc_r) / C_vc
            
        Tmax, tau, trans = base.Tmax_lv, base.tau_lv, trans_lv
        e_lv = torch.where(t <= trans,
                    0.5 * (1 - torch.cos(np.pi / Tmax * t)),
                    0.5 * (1 - torch.cos(np.pi / Tmax * trans)) * \
                        torch.exp((-t + trans) / tau))
        A, B, v_r = base.A_lv, base.B_lv, base.v_lv_r
        p_lv = e_lv * Ees_lv * (v_lv_p - v_r) + (1 - e_lv) * A * (
            torch.exp(B * (v_lv_p - v_r)) - 1)
        
        
        t_la = torch.where(t<700, t + 100, t + 100 - Tc)
        Tmax, tau, trans = base.Tmax_la, base.tau_la, base.trans_la
        e_la = torch.where(t_la <= trans,
                    0.5 * (1 - torch.cos(np.pi / Tmax * t_la)),
                    0.5 * (1 - torch.cos(np.pi / Tmax * trans)) * \
                        torch.exp((-t_la + trans) / tau))
        A, B, v_r = base.A_la, base.B_la, base.v_la_r
        p_la = e_la * Ees_la * (v_la_p - v_r) + (1 - e_la) * A * (
            torch.exp(B * (v_la_p - v_r)) - 1)
        
        """ calculate the maximum of the pressure difference and zero using
        softplus with regularization """
        alpha = 0.01
        q_av = (1 / alpha) * nn.functional.softplus(alpha * (p_lv - p_ao)) / R_av
        q_mv = (1 / alpha) * nn.functional.softplus(alpha * (p_la - p_lv)) / R_mv
        
        q_ao = (p_ao - p_art) / R_ao
        q_art = (p_art - p_vc) / R_art
        q_vc = (p_vc - p_la) / R_vc

        if self.num_derivative:
            dnvdnt = NumericalDerivative(L=1.)
            dndntv_lv = dnvdnt(sV[:, 0:1])
            dndntv_ao = dnvdnt(sV[:, 1:2])
            dndntv_art = dnvdnt(sV[:, 2:3])
            dndntv_la = dnvdnt(sV[:, 3:4])

        else:
            dndntv_lv = torch.autograd.grad(
                sV[:, 0:1],
                nt,
                grad_outputs=torch.ones_like(sV[:, 0:1]),
                create_graph=True)[0][:, 0:1]
            
            dndntv_ao = torch.autograd.grad(
                sV[:, 1:2],
                nt,
                grad_outputs=torch.ones_like(sV[:, 1:2]),
                create_graph=True)[0][:, 0:1]
            
            dndntv_art = torch.autograd.grad(
                sV[:, 2:3],
                nt,
                grad_outputs=torch.ones_like(sV[:, 2:3]),
                create_graph=True)[0][:, 0:1]
            
            dndntv_la = torch.autograd.grad(
                sV[:, 3:4],
                nt,
                grad_outputs=torch.ones_like(sV[:, 3:4]),
                create_graph=True)[0][:, 0:1]
        
        sv_MIN = Scaling.sv_range[0]
        sv_MAX = Scaling.sv_range[1]
        ddtv_lv = (Scaling.MAX.v_lv - Scaling.MIN.v_lv) / (
            sv_MAX - sv_MIN) / Tc * dndntv_lv
        ddtv_ao = (Scaling.MAX.v_ao - Scaling.MIN.v_ao) / (
            sv_MAX - sv_MIN) / Tc * dndntv_ao
        ddtv_art = (Scaling.MAX.v_art - Scaling.MIN.v_art) / (
            sv_MAX - sv_MIN) / Tc * dndntv_art
        ddtv_la = (Scaling.MAX.v_la - Scaling.MIN.v_la) / (
            sv_MAX - sv_MIN) / Tc * dndntv_la
        ddtv_vc = - (ddtv_lv + ddtv_ao + ddtv_art + ddtv_la)

        ODE1 = ddtv_lv - (q_mv - q_av)
        ODE2 = ddtv_ao - (q_av - q_ao)
        ODE3 = ddtv_art - (q_ao - q_art)
        ODE4 = ddtv_vc - (q_art - q_vc)
        ODE5 = ddtv_la -(q_vc - q_mv)

        return torch.concat((ODE1, ODE2, ODE3, ODE4, ODE5), axis=1)


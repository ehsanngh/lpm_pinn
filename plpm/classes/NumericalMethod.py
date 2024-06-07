import torch
import numpy as np
from plpm.classes.processing import ModelProcessing
import time
from torch import nn
import sys

def e(t, Tmax, tau, trans):
    """
    The time-varying part of the left ventricle and the left atrium's
    elastance.

    Inputs:
    - t: time in (ms)
    - Tmax: the time to end-systole (time to peak)
    - tau: the relaxation time constant
    - trans: the time when transitioniing to diastole 
    """
    if t <= trans:
        out = 0.5 * (1 - torch.cos(np.pi / Tmax * t))
    else:
        coeff = 0.5 * (1 - torch.cos(np.pi / Tmax * trans))
        exp_term = torch.exp((-t + trans) / tau)
        out = coeff * exp_term
    return out


class NumericalSolver:
    def __init__(self,
                 model_processor: ModelProcessing):
        self.mp = model_processor
        self.parameter_update = self.mp.parameter_update

    def netflowrates(self, t, V, exact_max):
        v_lv, v_ao, v_art, v_vc, v_la = torch.split(V, 1, dim=1)
        p_ao = (v_ao - self.mp.v_ao_r) / self.mp.C_ao
        p_art = (v_art - self.mp.v_art_r) / self.mp.C_art
        p_vc = (v_vc - self.mp.v_vc_r) / self.mp.C_vc
        e_lv = e(t, self.mp.Tmax_lv, self.mp.tau_lv, self.mp.trans_lv)
        p_lv = e_lv * self.mp.Ees_lv * (
            v_lv - self.mp.v_lv_r) + (1 - e_lv) * self.mp.A_lv * (
            torch.exp(self.mp.B_lv * (v_lv - self.mp.v_lv_r)) - 1)
        if t < 700:
            t_la = t + 100
        else:
            t_la = t + 100 - self.mp.Tc
        e_la = e(t_la, self.mp.Tmax_la, self.mp.tau_la, self.mp.trans_la)
        p_la = e_la * self.mp.Ees_la * (
            v_la - self.mp.v_la_r) + (1 - e_la) * self.mp.A_la * (
            torch.exp(self.mp.B_la * (v_la - self.mp.v_la_r)) - 1)
        
        
        if exact_max:
            q_av = torch.maximum(p_lv - p_ao, torch.tensor(0.)) / self.mp.R_av
            q_mv = torch.maximum(p_la - p_lv, torch.tensor(0.)) / self.mp.R_mv
        else:
            """ calculate the maximum of the pressure difference and
            zero using softplus with regularization """
            alpha = 0.01
            q_av = (1 / alpha) * nn.functional.softplus(alpha * (p_lv - p_ao)
                                                        ) / self.mp.R_av
            q_mv = (1 / alpha) * nn.functional.softplus(alpha * (p_la - p_lv)
                                                        ) / self.mp.R_mv

        q_ao = (p_ao - p_art) / self.mp.R_ao
        q_art = (p_art - p_vc) / self.mp.R_art
        q_vc = (p_vc - p_la) / self.mp.R_vc

        Q = torch.concat((q_mv - q_av,
                          q_av - q_ao,
                          q_ao - q_art,
                          q_art - q_vc,
                          q_vc - q_mv), axis=1)
        return Q

    def __call__(self, CASE, dt=1, n_cycle=15, method='Euler',
                 exact_max=False, verbose=True):
        self.parameter_update(CASE)

        time_steps = torch.linspace(0, self.mp.Tc, int(self.mp.Tc/dt) + 1)
        v_lv = torch.zeros_like(time_steps)
        v_ao = torch.zeros_like(time_steps)
        v_art = torch.zeros_like(time_steps)
        v_vc = torch.zeros_like(time_steps)
        v_la = torch.zeros_like(time_steps)

        v_lv[0] = 0.02 * self.mp.V_total
        v_ao[0] = 0.025 * self.mp.V_total
        v_art[0] = 0.21 * self.mp.V_total
        v_vc[0] = 0.727 * self.mp.V_total
        v_la[0] = 0.018 * self.mp.V_total

        SS_error = torch.ones(5)
        start = time.time()
        for cycle in range(n_cycle):
            for i, t in enumerate(time_steps[:-1]):
                V = torch.hstack((v_lv[i],
                                  v_ao[i],
                                  v_art[i],
                                  v_vc[i],
                                  v_la[i])).view((1, -1))
                
                if method == 'Euler':
                    Q = self.netflowrates(t, V, exact_max)
                    V_updated = V + dt * Q
                
                elif method == 'RK4':
                    k1 = self.netflowrates(t, V, exact_max)
                    k2 = self.netflowrates(t + dt / 2,
                                           V + dt / 2 * k1,
                                           exact_max)
                    k3 = self.netflowrates(t + dt / 2,
                                           V + dt / 2 * k2,
                                           exact_max)
                    k4 = self.netflowrates(t + dt / 2,
                                           V + dt * k3,
                                           exact_max)

                    V_updated = V + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

                else:
                    print(f'Error! Method {method} is not implemented. Use \'Euler\' or \'RK4\'.')
                    sys.exit()


                v_lv[i + 1] = V_updated.flatten()[0]
                v_ao[i + 1] = V_updated.flatten()[1]
                v_art[i + 1] = V_updated.flatten()[2]
                v_vc[i + 1] = V_updated.flatten()[3]
                v_la[i + 1] = V_updated.flatten()[4]

            SS_error[0] = abs(v_lv[0] - v_lv[-1]) / v_lv[-1]
            SS_error[1] = abs(v_ao[0] - v_ao[-1]) / v_ao[-1]
            SS_error[2] = abs(v_art[0] - v_art[-1]) / v_art[-1]
            SS_error[3] = abs(v_vc[0] - v_vc[-1]) / v_vc[-1]
            SS_error[4] = abs(v_la[0] - v_la[-1]) / v_la[-1]

            if torch.all(SS_error < 1e-4):
                if verbose:
                    print('Steady State Solution')
                break

            v_lv[0], v_ao[0], v_art[0] = v_lv[-1], v_ao[-1], v_art[-1]
            v_vc[0], v_la[0] = v_vc[-1], v_la[-1]

        finish = time.time()
        if verbose:
            print('Elapsed Time: ', finish - start, 'seconds')
            
        return torch.vstack((v_lv, v_ao, v_art, v_vc, v_la)).T

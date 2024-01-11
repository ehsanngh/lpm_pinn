import torch
import numpy as np
from plpm.constants.BaselineCase import base
import time
from torch import nn

def e(t, Tmax, tau, trans):
    """
    The time-varying part of the left ventricle and the left atrium's
    elastance.

    Inputs:
    - t: time in (ms)
    - Tmax: the time to end-systole
    - tau: the relaxation time constant
    - trans: the time when transiting to diastole 
    """
    if t <= trans:
        out = 0.5 * (1 - torch.cos(np.pi / Tmax * t))
    else:
        coeff = 0.5 * (1 - torch.cos(np.pi / Tmax * trans))
        exp_term = torch.exp((-t + trans) / tau)
        out = coeff * exp_term
    return out


def calculate_PLV(v_lv, mEes_lv=torch.tensor(1.), mtrans_lv=torch.tensor(1.)):
    t = torch.linspace(0., 1., len(v_lv)) * base.Tc
    Ees_lv = base.Ees_lv * mEes_lv
    trans_lv = base.trans_lv * mtrans_lv
    Tmax, tau, trans = base.Tmax_lv, base.tau_lv, trans_lv
    e_lv = torch.where(t <= trans,
                       0.5 * (1 - torch.cos(np.pi / Tmax * t)),
                       0.5 * (1 - torch.cos(np.pi / Tmax * trans)) * \
                        torch.exp((-t + trans) / tau))
    A, B, v_r = base.A_lv, base.B_lv, base.v_lv_r
    p_lv = e_lv * Ees_lv * (v_lv - v_r) + (1 - e_lv) * A * (
    torch.exp(B * (v_lv - v_r)) - 1)
    return p_lv * 0.0075


def numerical_solver(CASE, dt=1, n_cycle = 15, verbose=1):
    mR_av, mR_ao, mC_ao, mR_art, mC_art, mR_vc, mC_vc, mR_mv, mEes_lv, mtrans_lv = torch.split(CASE, 1, dim=1)
    
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
    V_total = base.V_total

    time_steps = torch.linspace(0, Tc, int(Tc/dt) + 1)
    v_lv = torch.zeros_like(time_steps)
    v_ao = torch.zeros_like(time_steps)
    v_art = torch.zeros_like(time_steps)
    v_vc = torch.zeros_like(time_steps)
    v_la = torch.zeros_like(time_steps)
    p_lv = torch.zeros_like(time_steps)

    v_lv[0] = 0.02 * V_total
    v_ao[0] = 0.025 * V_total
    v_art[0] = 0.21 * V_total
    v_vc[0] = 0.727 * V_total
    v_la[0] = 0.018 * V_total

    SS_error = torch.ones(5)

    start = time.time()
    for cycle in range(n_cycle):
        for i, t in enumerate(time_steps[:-1]):
            p_ao = (v_ao[i] - base.v_ao_r) / C_ao
            p_art = (v_art[i] - base.v_art_r) / C_art
            p_vc = (v_vc[i] - base.v_vc_r) / C_vc
            e_lv = e(t, base.Tmax_lv, base.tau_lv, trans_lv)
            p_lv[i + 1] = e_lv * Ees_lv * (
                v_lv[i] - base.v_lv_r) + (1 - e_lv) * base.A_lv * (
                torch.exp(base.B_lv * (v_lv[i] - base.v_lv_r)) - 1)
            if t < 700:
                t_la = t + 100
            else:
                t_la = t + 100 - Tc
            e_la = e(t_la, base.Tmax_la, base.tau_la, base.trans_la)
            p_la = e_la * Ees_la * (
                v_la[i] - base.v_la_r) + (1 - e_la) * base.A_la * (
                torch.exp(base.B_la * (v_la[i] - base.v_la_r)) - 1)
            
            """ calculate the maximum of the pressure difference and
            zero using softplus with regularization """
            alpha = 0.01
            q_av = (1 / alpha) * nn.functional.softplus(
                alpha * (p_lv[i + 1] - p_ao)) / R_av
            q_mv = (1 / alpha) * nn.functional.softplus(
                alpha * (p_la - p_lv[i + 1])) / R_mv
            
            q_ao = (p_ao - p_art) / R_ao
            q_art = (p_art - p_vc) / R_art
            q_vc = (p_vc - p_la) / R_vc

            v_lv[i + 1] = v_lv[i] + dt * (q_mv - q_av)
            v_ao[i + 1] = v_ao[i] + dt * (q_av - q_ao)
            v_art[i + 1] = v_art[i] + dt * (q_ao - q_art)
            v_vc[i + 1] = v_vc[i] + dt * (q_art - q_vc)
            v_la[i + 1] = v_la[i] + dt * (q_vc - q_mv)
        
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
        p_lv[0] = p_lv[-1]
    finish = time.time()
    if verbose:
        print('Elapsed Time: ', finish - start, 'seconds')
        
    return torch.vstack((v_lv, v_ao, v_art, v_vc, v_la)).T

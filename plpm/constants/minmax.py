import torch
from plpm.constants.BaselineCase import base

m_min, m_max = torch.tensor([.3, 3.])
class minValues:
    def __init__(self):
        self.mR_av = m_min
        self.mR_ao = m_min
        self.mC_ao = m_min
        self.mR_art = m_min
        self.mC_art = m_min
        self.mR_vc = m_min
        self.mC_vc = m_min
        self.mR_mv = m_min
        self.mEes_lv = m_min
        self.mtrans_lv = torch.tensor(0.8)

        self.v_lv = base.v_lv_r
        self.v_ao = base.v_ao_r
        self.v_art = base.v_art_r
        self.v_la = base.v_la_r


class maxValues:
    def __init__(self):
        self.mR_av = m_max
        self.mR_ao = m_max
        self.mC_ao = m_max
        self.mR_art = m_max
        self.mC_art = m_max
        self.mR_vc = m_max
        self.mC_vc = m_max
        self.mR_mv = m_max
        self.mEes_lv = m_max
        self.mtrans_lv = torch.tensor(1.2)

        self.v_lv = torch.tensor(150.)
        self.v_ao = torch.tensor(250.)
        self.v_art = torch.tensor(1800.)
        self.v_la = torch.tensor(110.)



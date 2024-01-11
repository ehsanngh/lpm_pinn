import torch

class BaselineCase:
    def __init__(self):
    
        self.V_total = torch.tensor(5200.0)  # ml
        self.R_av = torch.tensor(800.)  # Aortic Valve Pa.ms.ml^(-1)


        self.R_ao = torch.tensor(32000.)  # Aorta
        self.C_ao = torch.tensor(0.0025)  # ml.Pa^(-1)

        self.R_art = torch.tensor(150000.)  # Arteries
        self.C_art = self.C_ao * 10

        self.R_vc = torch.tensor(1200.)  # Vena Cava
        self.C_vc = torch.tensor(1.)

        self.R_mv = torch.tensor(550.)  # Mitral Valve

        self.Tc = torch.tensor(800.)  # Cycle time in ms

        # Resting Volumes
        self.v_ao_r = torch.tensor(100.)
        self.v_art_r = torch.tensor(900.)
        self.v_vc_r = torch.tensor(2800.)

        # LV Parameters
        self.Ees_lv = torch.tensor(400.)  # Left Ventricle Contractility
        self.A_lv = torch.tensor(1/0.0075)
        self.B_lv = torch.tensor(0.027)
        self.v_lv_r = torch.tensor(10.0)
        self.Tmax_lv = torch.tensor(280.0)
        self.tau_lv = torch.tensor(25.0)
        self.trans_lv = 1.5 * self.Tmax_lv

        # LA Parameters
        self.Ees_la = torch.tensor(60.)
        self.A_la = torch.tensor(0.44/0.0075) 
        self.B_la = torch.tensor(0.05)
        self.v_la_r = torch.tensor(10)
        self.Tmax_la = torch.tensor(150)
        self.tau_la = torch.tensor(25)
        self.trans_la = 1.5 * self.Tmax_la


base = BaselineCase()

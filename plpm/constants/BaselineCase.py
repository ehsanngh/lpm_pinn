import torch

m_min, m_max = torch.tensor([.3, 3.0])
parameters = {
    'V_total': {'base': torch.tensor(5200.0),  # ml
                'is_input': False,
                'm_min': m_min,
                'm_max': m_max},
    'R_av': {'base': torch.tensor(800.),  # Aortic Valve Pa.ms.ml^(-1)
             'is_input': False,
             'm_min': m_min,
             'm_max': m_max}, 
    
    'R_ao': {'base': torch.tensor(32000.),  # Aorta
             'is_input': False,
             'm_min': m_min,
             'm_max': m_max},  
    'C_ao': {'base': torch.tensor(0.0025),  # ml.Pa^(-1)
             'is_input': False,
             'm_min': m_min,
             'm_max': m_max},
    
    'R_art': {'base': torch.tensor(150000.),  # Arteries
              'is_input': False,
              'm_min': m_min,
              'm_max': m_max},
    'C_art': {'base': torch.tensor(0.0025) * 10,
              'is_input': False,
              'm_min': m_min,
              'm_max': m_max},
    
    'R_vc': {'base': torch.tensor(1200.),  # Vena Cava
             'is_input': False,
             'm_min': m_min,
             'm_max': m_max},
    'C_vc': {'base': torch.tensor(1.),
             'is_input': False,
             'm_min': m_min,
             'm_max': m_max},

    'R_mv': {'base': torch.tensor(550.),  # Mitral Valve
             'is_input': False,
             'm_min': m_min,
             'm_max': m_max},

    'Tc': {'base': torch.tensor(800.),  # Cycle time in ms
           'is_input': False,
           'm_min': m_min,
           'm_max': m_max},

    # Resting Volumes
    'v_ao_r': {'base': torch.tensor(100.),
               'is_input': False,
               'm_min': m_min,
               'm_max': m_max},
    'v_art_r': {'base': torch.tensor(900.),
                'is_input': False,
                'm_min': m_min,
                'm_max': m_max},
    'v_vc_r': {'base': torch.tensor(2800.),
               'is_input': False,
               'm_min': m_min,
               'm_max': m_max},

    # LV Parameters
    'Ees_lv': {'base': torch.tensor(400.),  # LV Contractility Pa.ml^(-1)
               'is_input': False,
               'm_min': m_min,
               'm_max': m_max},
    'A_lv': {'base': torch.tensor(1/0.0075),
             'is_input': False,
             'm_min': m_min,
             'm_max': m_max},
    'B_lv': {'base': torch.tensor(0.027),
             'is_input': False,
             'm_min': m_min,
             'm_max': m_max},
    'v_lv_r': {'base': torch.tensor(10.0),
               'is_input': False,
               'm_min': m_min,
               'm_max': m_max},
    'Tmax_lv': {'base': torch.tensor(280.0),
                'is_input': False,
                'm_min': torch.tensor(.8),
                'm_max': torch.tensor(1.2)},
    'tau_lv': {'base': torch.tensor(25.0),
               'is_input': False,
               'm_min': m_min,
               'm_max': m_max},
    'trans_lv': {'base': 1.5 * torch.tensor(280.0),
                 'is_input': False,
                 'm_min': torch.tensor(0.8),
                 'm_max': torch.tensor(1.2)},

    # LA Parameters
    'Ees_la': {'base': torch.tensor(60.),
               'is_input': False,
               'm_min': m_min,
               'm_max': m_max},
    'A_la': {'base': torch.tensor(0.44/0.0075),
             'is_input': False,
             'm_min': m_min,
             'm_max': m_max}, 
    'B_la': {'base': torch.tensor(0.05),
             'is_input': False,
             'm_min': m_min,
             'm_max': m_max},
    'v_la_r': {'base': torch.tensor(10),
               'is_input': False,
               'm_min': m_min,
               'm_max': m_max},
    'Tmax_la': {'base': torch.tensor(150),
                'is_input': False,
                'm_min': m_min,
                'm_max': m_max},
    'tau_la': {'base': torch.tensor(25),
               'is_input': False,
               'm_min': m_min,
               'm_max': m_max},
    'trans_la': {'base': 1.5 * torch.tensor(150),
                 'is_input': False,
                 'm_min': m_min,
                 'm_max': m_max},
}
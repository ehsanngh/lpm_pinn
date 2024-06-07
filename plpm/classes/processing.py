import torch

class ModelProcessing:
    """
    This class performs essential pre-processing and post-processing
    tasks, including interpreting the model input parameters, performing
    scaling and inverse scaling on model inputs and outputs, and updating
    model parameter values at different stages.

    Inputs to initialize:
    - parameters: A dictionary of model parameters. This dictionary is
    saved in the constants folder with the attribute 'is_input' set to
    False for all inputs. This dictionary should be updated by one of
    the config files (or manually) before being passed here.
    - Vs_min: The minimum expected volumes for V_LV, V_ao, V_art, and
    V_la across all cases. It must be a torch.tensor of shape (1, 4).
    - Vs_max: Similar to Vs_min, but for maximum volumes. Vs_min and
    Vs_max have some predefined values and are optional inputs.
    - sv_range: The range used to scale the volumes. By default, the
    scaled volumes are scaled between -1 and 1.
    """
    def __init__(self,
                 parameters,
                 Vs_min=None,
                 Vs_max=None,
                 sv_range=torch.tensor([-1., 1.]),
                 verbose=False):
        """
        - self.d (int): The number of physiological input parameters
        - self.minmax (torch.tensor): The minimum and maximum values for each
        input parameter. Initially, it is defined as a list. However, at the
        end of the initialization function, it will be converted to a PyTorch
        tensor of shape (2, self.d), where the upper and lower rows correspond
        to the minimum and maximum values of each input, respectively.
        - self.input_decoder (callable): This function decodes the inputs
        to update the model parameter values accordingly. Currently, the
        assignment of the appropriate decoder to each model is based on the
        number of inputs. This means that it is not possible to have, for
        example, two different decoders for models that have two different
        sets of 5 model parameters as inputs. To try different model
        parameters, all that needs to be done is to ensure that the correct
        decoder is defined and assigned to the model.
        """
        self.d = 0 
        self.minmax = []
        for param, values in parameters.items():
            if values.get('is_input') == True:
                self.d += 1
                self.minmax.append((values.get('m_min'), values.get('m_max')))
                if verbose:
                    print(f'Input parameter {self.d}: {param}')
        if Vs_min == None:
            self.Vs_min = torch.tensor((parameters['v_lv_r']['base'],
                                        parameters['v_ao_r']['base'],
                                        parameters['v_art_r']['base'],
                                        parameters['v_la_r']['base'])).view(1, 4)
        if Vs_max == None:
            self.Vs_max = torch.tensor((150., 250., 1800., 110.)).view(1, 4)               
        
        self.sv_range = sv_range
        self.minmax = torch.tensor(self.minmax).T
        self.pmtrs = parameters

        self.multipliers_initialization()
        
        if self.d == 0:
            self.input_decoder = lambda x: None
            self.minmax = torch.randn((2, 2))
        elif self.d == 2:
            self.input_decoder = self.input_decoder2Dmodel
        elif self.d == 4:
            self.input_decoder = self.input_decoder4Dmodel
        elif self.d == 5:
            self.input_decoder = self.input_decoder5Dmodel
        elif self.d == 10:
            self.input_decoder = self.input_decoder10Dmodel

    def multipliers_initialization(self):
        """
        This function initializes the multipliers for all physiological
        parameters to one. Any physiological parameters that are model
        inputs will be updated accordingly later.
        """
        self.mV_total = torch.tensor(1.)
        self.mR_av, self.mR_ao, self.mC_ao, self.mR_art, self.mC_art = torch.ones(5)
        self.mR_vc, self.mC_vc, self.mR_mv, self.mTc = torch.ones(4)
        self.mv_ao_r, self.mv_art_r, self.mv_vc_r = torch.ones(3)
        self.mEes_lv, self.mA_lv, self.mB_lv, self.mv_lv_r = torch.ones(4)
        self.mTmax_lv, self.mtau_lv, self.mtrans_lv = torch.ones(3)
        self.mEes_la, self.mA_la, self.mB_la, self.mv_la_r = torch.ones(4)
        self.mTmax_la, self.mtau_la, self.mtrans_la = torch.ones(3)

    def input_decoder2Dmodel(self, CASE):
        self.mEes_lv, self.mtrans_lv = torch.split(CASE, 1, dim=1)

    def input_decoder4Dmodel(self, CASE):
        self.mC_art, self.mC_vc, self.mEes_lv, \
            self.mtrans_lv = torch.split(CASE, 1, dim=1)

    def input_decoder5Dmodel(self, CASE):
        self.mC_art, self.mC_vc, self.mEes_lv, self.mTmax_lv, \
            self.mtrans_lv = torch.split(CASE, 1, dim=1)
    
    def input_decoder10Dmodel(self, CASE):
        self.mR_av, self.mR_ao, self.mC_ao, self.mR_art, self.mC_art, \
            self.mR_vc, self.mC_vc, self.mR_mv, self.mEes_lv, \
                self.mtrans_lv = torch.split(CASE, 1, dim=1)
    
    def parameter_update(self, CASE):
        """
        Given CASE, this function updates the values of the model parameters
        """
        self.input_decoder(CASE)
        self.V_total = self.pmtrs['V_total']['base'] * self.mV_total
        self.R_av = self.pmtrs['R_av']['base'] * self.mR_av
        self.R_ao = self.pmtrs['R_ao']['base'] * self.mR_ao
        self.C_ao = self.pmtrs['C_ao']['base'] * self.mC_ao
        self.R_art = self.pmtrs['R_art']['base'] * self.mR_art
        self.C_art = self.pmtrs['C_art']['base'] * self.mC_art
        self.R_vc = self.pmtrs['R_vc']['base'] * self.mR_vc
        self.C_vc = self.pmtrs['C_vc']['base'] * self.mC_vc
        self.R_mv = self.pmtrs['R_mv']['base'] * self.mR_mv
        self.Tc = self.pmtrs['Tc']['base'] * self.mTc
        self.v_ao_r = self.pmtrs['v_ao_r']['base'] * self.mv_ao_r
        self.v_art_r = self.pmtrs['v_art_r']['base'] * self.mv_art_r
        self.v_vc_r = self.pmtrs['v_vc_r']['base'] * self.mv_vc_r
        self.Ees_lv = self.pmtrs['Ees_lv']['base'] * self.mEes_lv
        self.A_lv = self.pmtrs['A_lv']['base'] * self.mA_lv
        self.B_lv = self.pmtrs['B_lv']['base'] * self.mB_lv
        self.v_lv_r = self.pmtrs['v_lv_r']['base'] * self.mv_lv_r
        self.Tmax_lv = self.pmtrs['Tmax_lv']['base'] * self.mTmax_lv
        self.tau_lv = self.pmtrs['tau_lv']['base'] * self.mtau_lv
        self.trans_lv = self.pmtrs['trans_lv']['base'] * self.mtrans_lv
        self.Ees_la = self.pmtrs['Ees_la']['base'] * self.mEes_la
        self.A_la = self.pmtrs['A_la']['base'] * self.mA_la
        self.B_la = self.pmtrs['B_la']['base'] * self.mB_la
        self.v_la_r = self.pmtrs['v_la_r']['base'] * self.mv_la_r
        self.Tmax_la = self.pmtrs['Tmax_la']['base'] * self.mTmax_la
        self.tau_la = self.pmtrs['tau_la']['base'] * self.mtau_la
        self.trans_la = self.pmtrs['trans_la']['base'] * self.mtrans_la

    def calculate_PLV(self, v_lv, CASE):
        """
        This function calculates P_LV and is used in post-processing.

        Inputs:
        - v_lv (torch.tensor): The LV volume waveform (ml), preferably
        in the shape of (num_timepoints, 1).
        - CASE (torch.tensor): A set of multipliers for the model inputs
        that correspond to the v_lv, in the shape of (1, self.d).

        Output:
        - p_lv (torch.tensor): The LV pressure waveform (mmHg), which
        has the same shape as v_lv.
        """
        self.parameter_update(CASE)
        t = (torch.linspace(0., 1., len(v_lv)) * self.Tc).view(v_lv.shape)
        Tmax, tau, trans = self.Tmax_lv, self.tau_lv, self.trans_lv
        e_lv = torch.where(t <= trans,
                        0.5 * (1 - torch.cos(torch.pi / Tmax * t)),
                        0.5 * (1 - torch.cos(torch.pi / Tmax * trans)) * \
                            torch.exp((-t + trans) / tau)).view(v_lv.shape)
        p_lv = e_lv * self.Ees_lv * (v_lv - self.v_lv_r) + (1 - e_lv) * \
            self.A_lv * (torch.exp(self.B_lv * (v_lv - self.v_lv_r)) - 1)
        return p_lv.view(v_lv.shape) * 0.0075
    
    def CaseScaler(self, CASE):
        """
        Scales the input case between zero and one.
        """
        self.to(CASE.device)
        sCASE = (CASE - self.minmax[0:1, :]) / \
            (self.minmax[1:2, :] - self.minmax[0:1, :])
        return sCASE


    def InvCaseScaler(self, sCASE):
        """
        Returns the scaled case, which is between zero and one, to its
        actual range.
        """
        self.to(sCASE.device)
        CASE = self.minmax[0:1, :] + sCASE * \
            (self.minmax[1:2, :] - self.minmax[0:1, :])
        return CASE

    def VolScaler(self, V):
        """
        Scales the volumes between sv_range[0] to sv_range[1].

        Input:
        - V: a tensor of size [:, 5] representing the volume of each
        compartment.

        Output:
        - sV: a tensor of size [:, 4] representing the scaled volumes of
        all compartments except v_vc.
        """
        self.to(V.device)
        V = torch.cat((V[:, :3], V[:, 4:]), dim=1)
        sV = (self.sv_range[1] - self.sv_range[0]) / \
            (self.Vs_max - self.Vs_min) * (V - self.Vs_min) + self.sv_range[0]
        return sV

    def InvVolScaler(self, sV, V_total=None):
        """
        Returns the scaled volumes, which is between sv_range[0] to
        sv_range[1], to their actual ranges.

        Input:
        - sV: a tensor of size [:, 4] representing the scaled
        volumes of all compartments except v_vc.

        Output:
        - V: a tensor of size [:, 5] representing the volume of each
        compartment. 
        """
        self.to(sV.device)
        if V_total is None:
            V_total = self.V_total
        V = (self.Vs_max - self.Vs_min) /\
            (self.sv_range[1] - self.sv_range[0]) * \
                (sV - self.sv_range[0]) + self.Vs_min
        v_vc_p = V_total - V.sum(dim=1, keepdim=True)
        return torch.cat((V[:, :3], v_vc_p, V[:, 3:]), dim=1)
    
    def to(self, device):
        self.minmax = self.minmax.to(device)
        self.sv_range = self.sv_range.to(device)
        self.Vs_max = self.Vs_max.to(device)
        self.Vs_min = self.Vs_min.to(device)

import torch
from plpm.constants.BaselineCase import base
from plpm.constants.minmax import minValues, maxValues


class scaling:
    def __init__(self, d=10, sv_range=torch.tensor([-1., 1.])):
        self.d = d
        self.sv_range = sv_range
        self.MIN = minValues()
        self.MAX = maxValues()

    def CaseScaler(self, CASE):
        """
        Scales the input case between zero and one.
        """
        MIN = self.MIN
        MAX = self.MAX

        if self.d == 4:
            mR_art, mC_art, mC_vc, mEes_lv = torch.split(CASE, 1, dim=1)

            nmR_art = (mR_art - MIN.mR_art) / (MAX.mR_art - MIN.mR_art)
            nmC_art = (mC_art - MIN.mC_art) / (MAX.mC_art - MIN.mC_art)
            nmC_vc = (mC_vc - MIN.mC_vc) / (MAX.mC_vc - MIN.mC_vc)
            nmEes_lv = (mEes_lv - MIN.mEes_lv) / (MAX.mEes_lv - MIN.mEes_lv)

            nCASE = torch.hstack((nmR_art, nmC_art, nmC_vc, nmEes_lv))

        elif self.d == 6:
            mR_art, mC_art, mC_vc, mR_mv, mEes_lv, \
                mtrans_lv = torch.split(CASE, 1, dim=1)

            nmR_art = (mR_art - MIN.mR_art) / (MAX.mR_art - MIN.mR_art)
            nmC_art = (mC_art - MIN.mC_art) / (MAX.mC_art - MIN.mC_art)
            nmC_vc = (mC_vc - MIN.mC_vc) / (MAX.mC_vc - MIN.mC_vc)
            nmR_mv = (mR_mv - MIN.mR_mv) / (MAX.mR_vc - MIN.mR_mv)
            nmEes_lv = (mEes_lv - MIN.mEes_lv) / (MAX.mEes_lv - MIN.mEes_lv)
            nmtrans_lv = (mtrans_lv - MIN.mtrans_lv) / (
                MAX.mtrans_lv - MIN.mtrans_lv)
            nCASE = torch.hstack((nmR_art, nmC_art, nmC_vc, nmR_mv,
                                  nmEes_lv, nmtrans_lv))
        
        elif self.d == 8:
            mR_ao, mC_ao, mR_art, mC_art, mC_vc, mR_mv, mEes_lv, \
                mtrans_lv = torch.split(CASE, 1, dim=1)

            nmR_ao = (mR_ao - MIN.mR_ao) / (MAX.mR_ao - MIN.mR_ao)
            nmC_ao = (mC_ao - MIN.mC_ao) / (MAX.mC_ao - MIN.mC_ao)

            nmR_art = (mR_art - MIN.mR_art) / (MAX.mR_art - MIN.mR_art)
            nmC_art = (mC_art - MIN.mC_art) / (MAX.mC_art - MIN.mC_art)

            nmC_vc = (mC_vc - MIN.mC_vc) / (MAX.mC_vc - MIN.mC_vc)

            nmR_mv = (mR_mv - MIN.mR_mv) / (MAX.mR_vc - MIN.mR_mv)

            nmEes_lv = (mEes_lv - MIN.mEes_lv) / (MAX.mEes_lv - MIN.mEes_lv)

            nmtrans_lv = (mtrans_lv - MIN.mtrans_lv) / (
                MAX.mtrans_lv - MIN.mtrans_lv)


            nCASE = torch.hstack((nmR_ao, nmC_ao, nmR_art, nmC_art,
                                  nmC_vc, nmR_mv, nmEes_lv, nmtrans_lv))
            
        elif self.d == 10:
            mR_av, mR_ao, mC_ao, mR_art, mC_art, mR_vc, mC_vc, mR_mv, \
                mEes_lv, mtrans_lv = torch.split(CASE, 1, dim=1)

            nmR_av = (mR_av - MIN.mR_av) / (MAX.mR_av - MIN.mR_av)

            nmR_ao = (mR_ao - MIN.mR_ao) / (MAX.mR_ao - MIN.mR_ao)
            nmC_ao = (mC_ao - MIN.mC_ao) / (MAX.mC_ao - MIN.mC_ao)

            nmR_art = (mR_art - MIN.mR_art) / (MAX.mR_art - MIN.mR_art)
            nmC_art = (mC_art - MIN.mC_art) / (MAX.mC_art - MIN.mC_art)

            nmR_vc = (mR_vc - MIN.mR_vc) / (MAX.mR_vc - MIN.mR_vc)
            nmC_vc = (mC_vc - MIN.mC_vc) / (MAX.mC_vc - MIN.mC_vc)

            nmR_mv = (mR_mv - MIN.mR_mv) / (MAX.mR_vc - MIN.mR_mv)

            nmEes_lv = (mEes_lv - MIN.mEes_lv) / (MAX.mEes_lv - MIN.mEes_lv)

            nmtrans_lv = (mtrans_lv - MIN.mtrans_lv) / (
                MAX.mtrans_lv - MIN.mtrans_lv)


            nCASE = torch.hstack((nmR_av, nmR_ao, nmC_ao, nmR_art, nmC_art,
                                    nmR_vc, nmC_vc, nmR_mv, nmEes_lv,
                                    nmtrans_lv))
            
        else:
            raise ValueError("Invalid value of d. \
                             Allowed values are 4, 6, 8, and 10.")
        return nCASE


    def InvCaseScaler(self, sCASE):
        """
        Returns the scaled case, which is between zero and one, to its
        actual range.
        """
        MIN = self.MIN
        MAX = self.MAX

        if self.d == 4:
            nmR_art, nmC_art, nmC_vc, nmEes_lv = torch.split(sCASE, 1, dim=1)

            mR_art = MIN.mR_art + nmR_art * (MAX.mR_art - MIN.mR_art)
            mC_art = MIN.mC_art + nmC_art * (MAX.mC_art - MIN.mC_art)
            mC_vc = MIN.mC_vc + nmC_vc * (MAX.mC_vc - MIN.mC_vc)
            mEes_lv = MIN.mEes_lv + nmEes_lv * (MAX.mEes_lv - MIN.mEes_lv)

            CASE = torch.hstack((mR_art, mC_art, mC_vc, mEes_lv))

        elif self.d == 6:
            nmR_art, nmC_art, nmC_vc, nmR_mv, nmEes_lv, \
                nmtrans_lv = torch.split(sCASE, 1, dim=1)


            mR_art = MIN.mR_art + nmR_art * (MAX.mR_art - MIN.mR_art)
            mC_art = MIN.mC_art + nmC_art * (MAX.mC_art - MIN.mC_art)

            mC_vc = MIN.mC_vc + nmC_vc * (MAX.mC_vc - MIN.mC_vc)

            mR_mv = MIN.mR_mv + nmR_mv * (MAX.mR_mv - MIN.mR_mv)

            mEes_lv = MIN.mEes_lv + nmEes_lv * (MAX.mEes_lv - MIN.mEes_lv)
            mtrans_lv = MIN.mtrans_lv + nmtrans_lv * (
                MAX.mtrans_lv - MIN.mtrans_lv)

            CASE = torch.hstack((mR_art, mC_art, mC_vc, mR_mv, mEes_lv,
                                 mtrans_lv))
        
        elif self.d == 8:
            nmR_ao, nmC_ao, nmR_art, nmC_art, nmC_vc, nmR_mv, nmEes_lv, \
                nmtrans_lv = torch.split(sCASE, 1, dim=1)

            mR_ao = MIN.mR_ao + nmR_ao * (MAX.mR_ao - MIN.mR_ao)
            mC_ao = MIN.mC_ao + nmC_ao * (MAX.mC_ao - MIN.mC_ao)

            mR_art = MIN.mR_art + nmR_art * (MAX.mR_art - MIN.mR_art)
            mC_art = MIN.mC_art + nmC_art * (MAX.mC_art - MIN.mC_art)

            mC_vc = MIN.mC_vc + nmC_vc * (MAX.mC_vc - MIN.mC_vc)

            mR_mv = MIN.mR_mv + nmR_mv * (MAX.mR_mv - MIN.mR_mv)

            mEes_lv = MIN.mEes_lv + nmEes_lv * (MAX.mEes_lv - MIN.mEes_lv)
            mtrans_lv = MIN.mtrans_lv + nmtrans_lv * (
                MAX.mtrans_lv - MIN.mtrans_lv)

            CASE = torch.hstack((mR_ao, mC_ao, mR_art, mC_art,
                                 mC_vc, mR_mv, mEes_lv, mtrans_lv))
            
        elif self.d == 10:
            nmR_av, nmR_ao, nmC_ao, nmR_art, nmC_art, nmR_vc, nmC_vc, \
                nmR_mv, nmEes_lv, nmtrans_lv = torch.split(sCASE, 1, dim=1)

            mR_av = MIN.mR_av + nmR_av * (MAX.mR_av - MIN.mR_av)

            mR_ao = MIN.mR_ao + nmR_ao * (MAX.mR_ao - MIN.mR_ao)
            mC_ao = MIN.mC_ao + nmC_ao * (MAX.mC_ao - MIN.mC_ao)

            mR_art = MIN.mR_art + nmR_art * (MAX.mR_art - MIN.mR_art)
            mC_art = MIN.mC_art + nmC_art * (MAX.mC_art - MIN.mC_art)

            mR_vc = MIN.mR_vc + nmR_vc * (MAX.mR_vc - MIN.mR_vc)
            mC_vc = MIN.mC_vc + nmC_vc * (MAX.mC_vc - MIN.mC_vc)

            mR_mv = MIN.mR_mv + nmR_mv * (MAX.mR_mv - MIN.mR_mv)

            mEes_lv = MIN.mEes_lv + nmEes_lv * (MAX.mEes_lv - MIN.mEes_lv)
            mtrans_lv = MIN.mtrans_lv + nmtrans_lv * (
                MAX.mtrans_lv - MIN.mtrans_lv)

            CASE = torch.hstack((mR_av, mR_ao, mC_ao, mR_art, mC_art,
                                    mR_vc, mC_vc, mR_mv, mEes_lv, mtrans_lv))
            
        else:
            raise ValueError("Invalid value of d. \
                             Allowed values are 4, 6, 8, and 10.")
            
        return CASE

    def VolScaler(self, V):
        """
        Scales the volumes between sv_range[0] to sv_range[1].

        Input:
        - V: a tensor of size [:, 5] representing the volume of each
        compartment.

        Output:
        - sV: a tensor of size [:, 4] representing the scaled volumes of all
        compartments except v_vc.
        """
        MIN = self.MIN
        MAX = self.MAX
        MIN_sv = self.sv_range[0]
        MAX_sv = self.sv_range[1]
        sv_lv = (MAX_sv - MIN_sv) / (
            MAX.v_lv - MIN.v_lv) * (V[:, 0:1] - MIN.v_lv) + MIN_sv
        sv_ao = (MAX_sv - MIN_sv) / (
            MAX.v_ao - MIN.v_ao) * (V[:, 1:2] - MIN.v_ao) + MIN_sv
        sv_art = (MAX_sv - MIN_sv) / (
            MAX.v_art - MIN.v_art) * (V[:, 2:3] - MIN.v_art) + MIN_sv
        sv_la = (MAX_sv - MIN_sv) / (
            MAX.v_la - MIN.v_la) * (V[:, 4:] - MIN.v_la) + MIN_sv
        return torch.hstack((sv_lv, sv_ao, sv_art, sv_la))

    def InvVolScaler(self, sV):
        """
        Returns the scaled volumes, which is between sv_range[0 ]to
        sv_range[1], to their actual ranges.

        Input:
        - sV: a tensor of size [:, 4] representing the scaled
        volumes of all compartments except v_vc.

        Output:
        - V: a tensor of size [:, 5] representing the volume of each
        compartment. 
        """
        MIN = self.MIN
        MAX = self.MAX
        MIN_sv = self.sv_range[0]
        MAX_sv = self.sv_range[1]
        v_lv_p = (MAX.v_lv - MIN.v_lv) / (
            MAX_sv - MIN_sv) * (sV[:, 0:1] - MIN_sv) + MIN.v_lv
        v_ao_p = (MAX.v_ao - MIN.v_ao) / (
            MAX_sv - MIN_sv) * (sV[:, 1:2] - MIN_sv) + MIN.v_ao
        v_art_p = (MAX.v_art - MIN.v_art) / (
            MAX_sv - MIN_sv) * (sV[:, 2:3] - MIN_sv) + MIN.v_art
        v_la_p = (MAX.v_la - MIN.v_la) / (
            MAX_sv - MIN_sv) * (sV[:, 3:4] - MIN_sv) + MIN.v_la
        v_vc_p = base.V_total - (v_lv_p + v_ao_p + v_art_p + v_la_p)
        return torch.hstack((v_lv_p, v_ao_p, v_art_p, v_vc_p, v_la_p))


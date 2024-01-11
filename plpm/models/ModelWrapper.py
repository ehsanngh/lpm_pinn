import torch
from plpm.classes.nn import MainPINN
from plpm.scripts_internal.GoverningEqns import GoverningEqns
from pkg_resources import resource_stream
torch.set_default_dtype(torch.float64)

class Model:
    def __init__(self,
                 model_file,
                 d,
                 sv_range=torch.tensor([-1., 1.]),
                 verbose=True):
        self.space_dim = d
        self.model_file = model_file
        self.sv_range = sv_range
        
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if verbose:
            print(self.device)
        
        torch.manual_seed(0)
        
        self.main_model = MainPINN(input_size=self.space_dim,
                                   hidden_size=256).to(self.device)
        main_snapshot = torch.load(resource_stream(
            'plpm', 'models/model/' + model_file + '.pt'),
            map_location=self.device)
        self.main_model.load_state_dict(main_snapshot["MODEL_STATE"])
        self.last_iter = main_snapshot["EPOCHS_RUN"]

        self.ODEs = GoverningEqns(d=self.space_dim,
                                  sv_range=self.sv_range)
        self.Scaling = self.ODEs.Scaling

    def _d_spredict_CASE(self, CASE, nt=None, num_timepoints=801):
        sCASE = self.Scaling.CaseScaler(CASE).to(self.device)

        if nt is None:
            nt = torch.linspace(
                0.0, 1.0, num_timepoints, dtype=CASE.dtype
                ).reshape((-1, 1)).requires_grad_().to(self.device)
        else:
            nt = nt.to(self.device)
        sCASE_repeated = torch.tile(sCASE, (len(nt), 1))
        sV_pinn = self.main_model(sCASE_repeated, nt)
        return nt, sCASE_repeated, sV_pinn

    def predict_CASE(self, CASE, nt=None, num_timepoints=801):
        _, _, sV_pinn = self._d_spredict_CASE(CASE, nt, num_timepoints)
        sV_pinn = sV_pinn.detach().cpu()
        V_pinn = self.Scaling.InvVolScaler(sV_pinn)
        return V_pinn
    
    def spredict_sCASE(self, sCASE, nt=None, num_timepoints=801):
        sCASE = sCASE.to(self.device)
        if nt is None:
            nt = torch.linspace(
                0.0, 1.0, num_timepoints, dtype=sCASE.dtype
                ).reshape((-1, 1)).requires_grad_().to(self.device)
        else:
            nt = nt.to(self.device)
        sCASE_repeated = torch.tile(sCASE, (len(nt), 1))
        sV_pinn = self.main_model(sCASE_repeated, nt)
        sV_pinn = sV_pinn.detach().cpu()
        return sV_pinn
    
    def predict_sCASE(self, sCASE, nt=None, num_timepoints=801):
        sV_pinn = self.spredict_sCASE(sCASE, nt, num_timepoints)
        V_pinn = self.Scaling.InvVolScaler(sV_pinn)
        return V_pinn

    def residuals_CASE(self, CASE, num_timepoints=801):
        nt, sCASE_repeated, sV_pinn = self._d_spredict_CASE(CASE,
                                                            num_timepoints)
        res = self.ODEs(sCASE_repeated, nt, sV_pinn).detach().cpu()
        return res


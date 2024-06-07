import torch
from plpm import Config, MainPINN, ModelProcessing
from plpm.scripts_internal.GoverningEqns import GoverningEqns
from pathlib import Path
current_path = Path(__file__).parent

torch.set_default_dtype(torch.float64)

class ModelWrapper:
    def __init__(self,
                 config: Config,
                 model_processor: ModelProcessing,
                 verbose=True):
        model_file = config.snapshot_path
        self.mp = model_processor
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if verbose:
            print(self.device)
        torch.manual_seed(0)
        self.main_model = MainPINN(input_size=self.mp.d,
                                   hidden_size=config.hidden_size,
                                   num_layers=config.num_layers).to(self.device)
        if model_file[-3:] != '.pt':
            model_file = current_path / 'model' / (model_file + '.pt')
            model_file = model_file.resolve()
        else:
            model_file = current_path / '..' / '..' /  model_file
            model_file = model_file.resolve()
        main_snapshot = torch.load(model_file, map_location=self.device)
        self.main_model.load_state_dict(main_snapshot["MODEL_STATE"])
        self.last_iter = main_snapshot["EPOCHS_RUN"]

        self.ODEs = GoverningEqns(model_processor=self.mp)

    def _d_spredict_CASE(self, CASE, nt=None, num_timepoints=801):
        self.mp.parameter_update(CASE)
        sCASE = self.mp.CaseScaler(CASE).to(self.device)

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
        V_pinn = self.mp.InvVolScaler(sV_pinn)
        return V_pinn
    
    def spredict_sCASE(self, sCASE, nt=None, num_timepoints=801):
        sCASE = sCASE.to(self.device)
        self.mp.parameter_update(self.mp.InvCaseScaler(sCASE))
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
        V_pinn = self.mp.InvVolScaler(sV_pinn)
        return V_pinn

    def residuals_CASE(self, CASE, num_timepoints=801):
        nt, sCASE_repeated, sV_pinn = self._d_spredict_CASE(CASE,
                                                            num_timepoints=num_timepoints)
        res = self.ODEs(sCASE_repeated, nt, sV_pinn).detach().cpu()
        return res


import torch
import numpy as np
import matplotlib.pyplot as plt
from plpm.constants.BaselineCase import parameters
from pathlib import Path
current_path = Path(__file__).parent


style_path = current_path / 'paper.mplstyle'
style_path = style_path.resolve()
plt.style.use(str(style_path))

Tc = parameters['Tc']['base']
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


class MainPlots:
    def __init__(self, dpi=300):
        plt.rcParams.update({'figure.dpi': dpi})
        None
    
    def plot_CASE_seperate(self,
                           V_pinn: torch.Tensor,
                           V_numerical: torch.Tensor=None):
        t_pinn = torch.linspace(0., 1., V_pinn.shape[0]) * Tc

        fig, ax = plt.subplots()
        ax.plot(t_pinn, V_pinn[:, 0], label='PINN Solution')
        ax.set_xlabel('$t \\: (ms)$')
        ax.set_ylabel('$V_{lv} \\: (ml)$')

        fig1, ax1 = plt.subplots()
        ax1.plot(t_pinn, V_pinn[:, 1], label='PINN Solution')
        ax1.set_xlabel('$t \\: (ms)$')
        ax1.set_ylabel('$V_{ao} \\: (ml)$')
        
        fig2, ax2 = plt.subplots()
        ax2.plot(t_pinn, V_pinn[:, 2], label='PINN Solution')
        ax2.set_xlabel('$t \\: (ms)$')
        ax2.set_ylabel('$V_{art} \\: (ml)$')

        fig3, ax3 = plt.subplots()
        ax3.plot(t_pinn, V_pinn[:, 3], label='PINN Solution')
        ax3.set_xlabel('$t \\: (ms)$')
        ax3.set_ylabel('$V_{vc} \\: (ml)$')
        
        fig4, ax4 = plt.subplots()
        ax4.plot(t_pinn, V_pinn[:, 4], label='PINN Solution')
        ax4.set_xlabel('$t \\: (ms)$')
        ax4.set_ylabel('$V_{la} \\: (ml)$')
        
        if V_numerical is not None:
            t_numerical = torch.linspace(0., 1., V_numerical.shape[0]) * Tc
            ax.plot(t_numerical, V_numerical[:, 0],
                    label='Numerical Solution', alpha=0.8, ls='-.')
            ax.legend()
            
            ax1.plot(t_numerical, V_numerical[:, 1],
                     label='Numerical Solution',alpha=0.8, ls='-.')
            ax1.legend()
            
            ax2.plot(t_numerical, V_numerical[:, 2],
                     label='Numerical Solution', alpha=0.8, ls='-.')
            ax2.legend()

            ax3.plot(t_numerical, V_numerical[:, 3],
                     label='Numerical Solution', alpha=0.8, ls='-.')
            ax3.legend()

            ax4.plot(t_numerical, V_numerical[:, 4],
                     label='Numerical Solution', alpha=0.8, ls='-.')
            ax4.legend()

        plt.show(fig)
        plt.show(fig1)
        plt.show(fig2)
        plt.show(fig3)
        plt.show(fig4)


    def plot_PVloop(self,
                 vlv_pinn: torch.Tensor,
                 plv_pinn: torch.Tensor,
                 vlv_numerical: torch.Tensor=None, 
                 plv_numerical: torch.Tensor=None):
        
        fig, ax = plt.subplots()
        ax.plot(vlv_pinn, plv_pinn, label='PINN Solution')
        ax.set_xlabel('$V_{lv} \\: (ml)$')
        ax.set_ylabel('$P_{lv} \\: (mmHg)$')
        if vlv_numerical is not None:
            ax.plot(vlv_numerical, plv_numerical, label='Numerical Solution',
                    alpha=0.9, ls='--')
            ax.legend()
        plt.show(fig)


    def plot_CASE_together(self,
                           V_pinn: torch.Tensor,
                           plv_pinn: torch.Tensor,
                           V_numerical: torch.Tensor=None,
                           plv_numerical: torch.Tensor=None):
        t_pinn = torch.linspace(0., 1., V_pinn.shape[0]) * Tc
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(9, 5.3))
        ax[0, 0].plot(t_pinn, V_pinn[:, 0], label='PINN Solution')
        ax[0, 0].set_xlabel('$t \\: (ms)$')
        ax[0, 0].set_ylabel('$V_{lv} \\: (ml)$')

        ax[0, 1].plot(t_pinn, V_pinn[:, 1], label='PINN Solution')        
        ax[0, 1].set_xlabel('$t \\: (ms)$')
        ax[0, 1].set_ylabel('$V_{ao} \\: (ml)$')

        ax[0, 2].plot(t_pinn, V_pinn[:, 2], label='PINN Solution')
        ax[0, 2].set_xlabel('$t \\: (ms)$')
        ax[0, 2].set_ylabel('$V_{art} \\: (ml)$')

        ax[1, 0].plot(t_pinn, V_pinn[:, 3], label='PINN Solution')
        ax[1, 0].set_xlabel('$t \\: (ms)$')
        ax[1, 0].set_ylabel('$V_{vc} \\: (ml)$')

        ax[1, 1].plot(t_pinn, V_pinn[:, 4], label='PINN Solution')
        ax[1, 1].set_xlabel('$t \\: (ms)$')
        ax[1, 1].set_ylabel('$V_{la} \\: (ml)$')

        ax[1, 2].plot(V_pinn[:, 0], plv_pinn, label='PINN Solution')
        
        ax[1, 2].set_xlabel('$V_{lv} \\: (ml)$')
        ax[1, 2].set_ylabel('$P_{lv} \\: (mmHg)$')

        
        if V_numerical is not None:
            t_numerical = torch.linspace(0., 1., V_numerical.shape[0]) * Tc
            ax[0, 0].plot(t_numerical, V_numerical[:, 0],
                          label='Numerical Solution', alpha=0.9, ls='--')
            ax[0, 1].plot(t_numerical, V_numerical[:, 1],
                          label='Numerical Solution', alpha=0.9, ls='--')
            ax[0, 2].plot(t_numerical, V_numerical[:, 2],
                          label='Numerical Solution', alpha=0.9, ls='--')
            ax[1, 0].plot(t_numerical, V_numerical[:, 3],
                          label='Numerical Solution', alpha=0.9, ls='--')
            ax[1, 1].plot(t_numerical, V_numerical[:, 4],
                          label='Numerical Solution', alpha=0.9, ls='--')
            ax[1, 2].plot(V_numerical[:, 0], plv_numerical,
                          label='Numerical Solution', alpha=0.9, ls='--')
            
            handles, labels = ax[1,2].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center',
                       edgecolor='black', facecolor='white',
                       ncol = 2, bbox_to_anchor=(0.48, .87))
        plt.show()

    
    def plot_residuals(self, residuals):
        t_plot = torch.linspace(0., 1., residuals.shape[0]) * Tc
        plt.plot(t_plot, residuals[:, 0], label='First ODE')
        plt.plot(t_plot, residuals[:, 1], label='Second ODE')
        plt.plot(t_plot, residuals[:, 2], label='Third ODE')
        plt.plot(t_plot, residuals[:, 3], label='Fourth ODE')
        plt.plot(t_plot, residuals[:, 4], label='Fifth ODE')
        plt.legend()
        plt.show()


    def plot_loss(self, loss_file):
        loss_file = current_path / '..' / '..' / (loss_file + '_Residuals.csv')
        loss_file = loss_file.resolve()
        loss = np.loadtxt(loss_file, delimiter=",", skiprows=1)
        epochs = loss[:, 0]
        train_loss = loss[:, 1]
        test_loss = loss[:, 2]
        fig, ax = plt.subplots()
        ax.plot(epochs, train_loss, label='Training Loss')
        ax.plot(epochs, test_loss, label='Test Loss')
        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        plt.show(fig)


#%%
class InverseModelingPlots:
    def __init__(self, data):
        self.num_signals = data.shape[1]
        self.t_data = data[:, 0]
        self.vlv_data = data[:, 1]
        if self.num_signals > 2:
            self.Plv_data = data[:, 2]
            if self.num_signals > 3:
                self.Pao_data = data[:, 3]
    
    def plot_VLV(self,
                 vlv_pinn: torch.Tensor,
                 vlv_num: torch.Tensor=None):
        t_pinn = torch.linspace(0., 1., vlv_pinn.shape[0]) * Tc

        fig, ax = plt.subplots()
        ax.plot(t_pinn, vlv_pinn, label='PINN Estimation')
        ax.plot(self.t_data, self.vlv_data, label='Data', alpha=0.9, ls='--')
        ax.set_xlabel('$t \\: (ms)$')
        ax.set_ylabel('$V_{lv} \\: (ml)$')
        
        if vlv_num is not None:
            t_numerical = torch.linspace(0., 1., vlv_num.shape[0]) * Tc
            ax.plot(t_numerical, vlv_num,
                    label='Numerical Solution', alpha=0.8, ls='--')
        ax.legend()
        plt.show(fig)

    def plot_PLV(self,
                 plv_pinn: torch.Tensor,
                 plv_num: torch.Tensor=None):
        t_pinn = torch.linspace(0., 1., plv_pinn.shape[0]) * Tc

        fig, ax = plt.subplots()
        ax.plot(t_pinn, plv_pinn, label='PINN Estimation')
        ax.plot(self.t_data, self.Plv_data, label='Data', alpha=0.9, ls='--')
        ax.set_xlabel('$t \\: (ms)$')
        ax.set_ylabel('$P_{lv} \\: (mmHg)$')
        
        if plv_num is not None:
            t_numerical = torch.linspace(0., 1., plv_num.shape[0]) * Tc
            ax.plot(t_numerical, plv_num,
                    label='Numerical Solution', alpha=0.8, ls='--')
        ax.legend()
        plt.show(fig)

    def plot_Pao(self,
                 Pao_pinn: torch.Tensor,
                 Pao_num: torch.Tensor=None):
        t_pinn = torch.linspace(0., 1., Pao_pinn.shape[0]) * Tc

        fig, ax = plt.subplots()
        ax.plot(t_pinn, Pao_pinn, label='PINN Estimation')
        ax.plot(self.t_data, self.Pao_data, label='Data', alpha=0.9, ls='--')
        ax.set_xlabel('$t \\: (ms)$')
        ax.set_ylabel('$P_{ao} \\: (mmHg)$')
        
        if Pao_num is not None:
            t_numerical = torch.linspace(0., 1., Pao_num.shape[0]) * Tc
            ax.plot(t_numerical, Pao_num,
                    label='Numerical Solution', alpha=0.8, ls='--')
        ax.legend()
        plt.show(fig)

    def plot_PVloop(self,
                    vlv_pinn: torch.Tensor,
                    plv_pinn: torch.Tensor,
                    metric: torch.Tensor=None,
                    metric_pos: torch.Tensor=torch.tensor([0.57, 0.93]),
                    vlv_numerical: torch.Tensor=None, 
                    plv_numerical: torch.Tensor=None):
        
        fig, ax = plt.subplots()
        ax.plot(vlv_pinn, plv_pinn, label='PINN Estimation')
        ax.plot(self.vlv_data, self.Plv_data, label='Data', alpha=0.9, ls='--')
        ax.set_xlabel('$V_{lv} \\: (ml)$')
        ax.set_ylabel('$P_{lv} \\: (mmHg)$')
        
        if vlv_numerical is not None:
            ax.plot(vlv_numerical, plv_numerical,
                    label='Numerical Solution', alpha=0.8, ls='--')
        ax.legend(loc='center')

        if metric is not None:
            ax.text(metric_pos[0], metric_pos[1], f'$R^{{2}}_{{v_{{lv}}}}$ = {metric[0]:.2f}, $R^{{2}}_{{p_{{lv}}}}$ = {metric[1]:.2f}',
                    transform=ax.transAxes)
            
        plt.show(fig)


    def plot_CASE_together(self,
                    vlv_pinn: torch.Tensor,
                    plv_pinn: torch.Tensor,
                    Pao_pinn: torch.Tensor=None,
                    vlv_num: torch.Tensor=None, 
                    plv_num: torch.Tensor=None,
                    Pao_num: torch.Tensor=None):
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9, 5.3))
        t_pinn = torch.linspace(0., 1., vlv_pinn.shape[0]) * Tc
        ax[0, 0].plot(t_pinn, vlv_pinn, label='PINN Estimation')
        ax[0, 0].plot(self.t_data, self.vlv_data, label='Data', alpha=0.9,
                      ls='--')
        ax[0, 0].set_xlabel('$t \\: (ms)$')
        ax[0, 0].set_ylabel('$V_{lv} \\: (ml)$')
        if vlv_num is not None:
            t_numerical = torch.linspace(0., 1., vlv_num.shape[0]) * Tc
            ax[0, 0].plot(t_numerical, vlv_num,
                    label='Numerical Solution', alpha=0.8, ls='--')
        
        t_pinn = torch.linspace(0., 1., plv_pinn.shape[0]) * Tc
        ax[0, 1].plot(t_pinn, plv_pinn, label='PINN Estimation')
        ax[0, 1].plot(self.t_data, self.Plv_data, label='Data', alpha=0.9, ls='--')
        ax[0, 1].set_xlabel('$t \\: (ms)$')
        ax[0, 1].set_ylabel('$P_{lv} \\: (mmHg)$')
        
        if plv_num is not None:
            t_numerical = torch.linspace(0., 1., plv_num.shape[0]) * Tc
            ax[0, 1].plot(t_numerical, plv_num,
                    label='Numerical Solution', alpha=0.8, ls='--')
   
        ax[1, 1].plot(vlv_pinn, plv_pinn, label='PINN Solution')
        ax[1, 1].plot(self.vlv_data, self.Plv_data, label='Data', alpha=0.9, ls='--')
        ax[1, 1].set_xlabel('$V_{lv} \\: (ml)$')
        ax[1, 1].set_ylabel('$P_{lv} \\: (mmHg)$')
        if vlv_num is not None:
            ax[1, 1].plot(vlv_num, plv_num, label='Numerical Solution', alpha=0.8, ls='--')

        if self.num_signals > 3:
            ax[1, 0].plot(t_pinn, Pao_pinn, label='PINN Estimation')
            ax[1, 0].plot(self.t_data, self.Pao_data, label='Data', alpha=0.9,
                          ls='--')
            ax[1, 0].set_xlabel('$t \\: (ms)$')
            ax[1, 0].set_ylabel('$P_{ao} \\: (mmHg)$')
            
            if Pao_num is not None:
                t_numerical = torch.linspace(0., 1., Pao_num.shape[0]) * Tc
                ax[1, 0].plot(t_numerical, Pao_num,
                              label='Numerical Solution', alpha=0.8, ls='--')
        
        handles, labels = ax[1,1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center',
                       edgecolor='black', facecolor='white',
                       ncol = 3, bbox_to_anchor=(0.48, .87))
        plt.show()

    def plot_publication(self,
                    vlv_pinn: torch.Tensor,
                    plv_pinn: torch.Tensor):
        
        fig, ax1 = plt.subplots(figsize=(3.5, 2.625))
        t_pinn = torch.linspace(0., 1., vlv_pinn.shape[0]) * Tc
        ax1.plot(t_pinn, vlv_pinn, alpha=0.75, label='$V_{PINN}$')
        ax1.plot(self.t_data, self.vlv_data, label='$V_{Data}$', alpha=1.,
                    ls='--', color= color_cycle[0])
        ax1.set_xlabel('$t \\: (ms)$')
        ax1.set_ylabel('$V_{lv} \\: (ml)$')
                
        ax2 = ax1.twinx()
        t_pinn = torch.linspace(0., 1., plv_pinn.shape[0]) * Tc
        ax2.plot(t_pinn, plv_pinn, color= color_cycle[1], alpha=0.75,
                 label='$P_{PINN}$')
        ax2.plot(self.t_data, self.Plv_data, color=color_cycle[1], alpha=1.,
                    label='$P_{Data}$', ls='--')
        ax2.set_ylabel('$P_{lv} \\: (mmHg)$')
        ax2.spines['right'].set_visible(True)
        ax2.spines['top'].set_visible(True)
                    
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(lines1 + lines2, labels1 + labels2,
                   loc='lower center',
                   edgecolor='black', facecolor='white',
                   ncol = 4, bbox_to_anchor=(0.48, .87))

        plt.show()


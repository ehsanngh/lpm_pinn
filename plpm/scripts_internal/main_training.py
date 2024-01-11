import torch
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader
from plpm import PINN_CP, MainPINN
from torch.optim.lr_scheduler import ReduceLROnPlateau
from plpm.scripts_internal.GoverningEqns import GoverningEqns

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Packages Required for Running the Model on Multiple GPUs ----------
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os


def setup():
    dist.init_process_group(backend="nccl")


# ----------------------------------------------------------------------------
def get_training_components(space_dim):
    torch.manual_seed(13370903)
    model = MainPINN(input_size=space_dim,
                     num_layers=5,
                     hidden_size=256,
                     activation_func = torch.nn.Tanh())
    learning_rate = 1e-3

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = 1e-3

    scheduler = ReduceLROnPlateau(optimizer,
                                  factor=0.5,
                                  patience=50,
                                  min_lr=1e-6)
    
    return model, optimizer, scheduler


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


# ----------------------------------------------------------------------------
class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            eqns: callable,
            nt: torch.Tensor,
            Train_DataLoader: DataLoader,
            Test_DataLoader: DataLoader,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            save_every: int,
            snapshot_path: str,
            loss_path: str,
            NumData_flag: bool,
            NumData_nt: torch.Tensor = None,
            NumData_DataLoader: DataLoader = None,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.eqns = eqns
        self.nt = nt.to(self.gpu_id)
        self.Train_DL = Train_DataLoader
        self.Test_DL = Test_DataLoader
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.scheduler = scheduler
        self.snapshot_path = snapshot_path
        self.loss_path = loss_path
        self.Residuals_epochs = []
        self.Residuals_TrainingLosses = []
        self.Residuals_TestLosses = []
        self.NumData_flag = NumData_flag
        if self.NumData_flag:
            self.NumData_nt = NumData_nt.to(self.gpu_id)
            self.NumData_DL = NumData_DataLoader
            self.NumData_epochs = []
            self.NumData_TrainingLosses = []
        
        if os.path.exists(self.snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(self.snapshot_path)

        if os.path.exists(self.loss_path + '_Residuals.csv'):
            Residuals_csv = np.loadtxt(self.loss_path + '_Residuals.csv',
                                       delimiter=",",
                                       skiprows=1)
            self.Residuals_epochs = Residuals_csv[:, 0]
            self.Residuals_TrainingLosses = Residuals_csv[:, 1]
            self.Residuals_TestLosses = Residuals_csv[:, 2]

            if self.NumData_flag:
                NumDataLosses_csv = np.loadtxt(
                    self.loss_path + '_NumericalDataLosses.csv',
                    delimiter=",",
                    skiprows=1)
                self.NumData_epochs = NumDataLosses_csv[:, 0]
                self.NumData_TrainingLosses = NumDataLosses_csv[:, 1]
            

        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER"])
        self.scheduler.load_state_dict(snapshot["SCHEDULER"])
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _numdata_batchloss(self, sCASEs_b, sV_b):
        num_cases = len(sCASEs_b)
        num_timepoints = len(self.NumData_nt)
        nt_repeated = torch.tile(self.NumData_nt, (num_cases, 1))  # nt repeated "num_cases" times
        sCASEs_repeated = sCASEs_b.repeat_interleave(num_timepoints, dim=0)
        sV_b = sV_b.view(num_timepoints * num_cases, 4)
        sV_pred = self.model(sCASEs_repeated, nt_repeated)
        batch_loss = torch.mean(torch.mean(torch.square(sV_pred - sV_b), dim=0))
        return batch_loss
    
    def _residuals_batchloss(self, sCASEs_b):
        num_cases = len(sCASEs_b)
        num_timepoints = len(self.nt)
        nt_repeated = torch.tile(self.nt, (num_cases, 1))  # nt repeated "num_cases" times
        sCASEs_repeated = sCASEs_b.repeat_interleave(num_timepoints, dim=0)
        sV_pred = self.model(sCASEs_repeated, nt_repeated)
        if self.optimizer.param_groups[0]['lr'] > 1.e-4:
            alpha = 0.002
            residuals = self.eqns(sCASEs_repeated, nt_repeated, sV_pred, alpha)
            batch_loss = torch.mean(torch.mean(torch.abs(residuals), dim=0))
        else:
            alpha = 0.01
            residuals = self.eqns(sCASEs_repeated, nt_repeated, sV_pred, alpha)
            batch_loss = torch.mean(torch.mean(torch.square(residuals), dim=0))
        return batch_loss

    def _epoch_training(self, epoch, w_res=0.9, w_numdata=0.1):
        residuals_bsz = len(next(iter(self.Train_DL)))
        self.Train_DL.sampler.set_epoch(epoch)
        residuals_totalloss = 0

        if self.NumData_flag:
            numdata_bsz = len(next(iter(self.NumericalData_DL))[0])
            self.NumData_DL.sampler.set_epoch(epoch)
            numdata_totalloss = 0
            for res_sCASEs_b, numdata_b in zip(self.Train_DL, self.NumData_DL):
                numdata_sCASEs_b, numdata_sV_b = numdata_b
                res_sCASEs_b = res_sCASEs_b.to(self.gpu_id)
                numdata_sCASEs_b, numdata_sV_b = numdata_sCASEs_b.to(self.gpu_id), numdata_sV_b.to(self.gpu_id)
                res_batchloss = self._residuals_batchloss(res_sCASEs_b)
                numdata_batchloss = self._numdata_batchloss(numdata_sCASEs_b, numdata_sV_b)
                batchloss_total = res_batchloss * w_res + numdata_batchloss * w_numdata
                self.optimizer.zero_grad()
                batchloss_total.backward(retain_graph=True)
                self.optimizer.step()

                res_batchloss = res_batchloss.item()
                numdata_batchloss = numdata_batchloss.item() 

                residuals_totalloss += res_batchloss * len(res_sCASEs_b)
                numdata_totalloss += numdata_batchloss * len(numdata_sCASEs_b)
            residuals_totalloss /= len(self.Train_DL.dataset)
            numdata_totalloss /= len(self.NumData_DL.dataset)
            totalloss = residuals_totalloss * w_res + numdata_totalloss * w_numdata
            print(f"Training: [GPU{self.gpu_id}] Epoch {epoch} | Batchsizes: ({residuals_bsz}, {numdata_bsz}) | Steps: ({len(self.Train_DL)}, {len(self.NumData_DL)}) | Lr: {self.optimizer.param_groups[0]['lr']} | Residuals Loss: {residuals_totalloss} | Data Loss: {numdata_totalloss}")
            output = (totalloss, residuals_totalloss, numdata_totalloss)
        else:
            for res_sCASEs_b in self.Train_DL:
                res_sCASEs_b = res_sCASEs_b.to(self.gpu_id)
                res_batchloss = self._residuals_batchloss(res_sCASEs_b)
                self.optimizer.zero_grad()
                res_batchloss.backward(retain_graph=True)
                self.optimizer.step()
                res_batchloss = res_batchloss.item()
                residuals_totalloss += res_batchloss * len(res_sCASEs_b)
            residuals_totalloss /= len(self.Train_DL.dataset)
            print(f"Training: [GPU{self.gpu_id}] Epoch {epoch} | Batchsize: ({residuals_bsz}) | Steps: ({len(self.Train_DL)}) | Lr: {self.optimizer.param_groups[0]['lr']} | Residuals Loss: {residuals_totalloss}")
            output = residuals_totalloss
        
        return output
    
    def _epoch_test(self, epoch):
        b_sz = len(next(iter(self.Test_DL)))
        self.Test_DL.sampler.set_epoch(epoch)
        totalloss = 0
        for sCASEs_b in self.Test_DL:
            sCASEs_b = sCASEs_b.to(self.gpu_id)
            batchloss = self._residuals_batchloss(sCASEs_b).item()
            totalloss += batchloss * len(sCASEs_b)

        totalloss /= len(self.Test_DL.dataset)
        print(f"Testing: [GPU{self.gpu_id}] Epoch {epoch} | Batchsizes: {b_sz} | Steps: {len(self.Test_DL)} | Lr: {self.optimizer.param_groups[0]['lr']} | Residuals Loss: {totalloss}")
        return totalloss

    def _save_snapshot(self, epoch):
        snapshot = {
            "EPOCHS_RUN": epoch,
            "MODEL_STATE": self.model.module.state_dict(),
            "OPTIMIZER": self.optimizer.state_dict(),
            "SCHEDULER": self.scheduler.state_dict(),
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at snapshot.pt")

    def _save_losses(self):
        np.savetxt(self.loss_path + '_Residuals.csv',
                   np.column_stack((self.Residuals_epochs,
                                    self.Residuals_TrainingLosses,
                                    self.Residuals_TestLosses,
                                    )),
                                    delimiter=",",
                                    header="Epoch, Train_Loss, Test_Loss")
        
        if self.NumData_flag:
            np.savetxt(self.loss_path + '_NumData.csv',
                       np.column_stack((self.NumData_epochs,
                                        self.NumData_TrainingLosses,
                                        )),
                                        delimiter=",",
                                        header="Epoch, Train_Loss")

    def train(self, max_epochs: int):
        timer = time.time()
        for epoch in range(self.epochs_run, max_epochs):
            if self.NumData_flag:
                totalloss, residuals_totalloss, numdata_totalloss = self._epoch_training(epoch)
                self.scheduler.step(totalloss)
                if self.gpu_id == 0 and (epoch + 1) % self.save_every == 0:
                    residuals_testloss = self._epoch_test(epoch)
                    self._save_snapshot(epoch + 1)
                    self.Residuals_epochs = np.append(self.Residuals_epochs,
                                                      epoch + 1)
                    self.NumData_epochs = np.append(self.NumData_epochs,
                                                    epoch + 1)
                    self.Residuals_TrainingLosses = np.append(
                        self.Residuals_TrainingLosses, residuals_totalloss)
                    self.Residuals_TestLosses = np.append(
                        self.Residuals_TestLosses, residuals_testloss)
                    self.NumData_TrainingLosses = np.append(
                        self.NumData_TrainingLosses, numdata_totalloss)
                    self._save_losses()
            else:
                residuals_totalloss = self._epoch_training(epoch)
                self.scheduler.step(residuals_totalloss)
                if self.gpu_id == 0 and (epoch + 1) % self.save_every == 0:
                    residuals_testloss = self._epoch_test(epoch)
                    self._save_snapshot(epoch + 1)
                    self.Residuals_epochs = np.append(self.Residuals_epochs,
                                                      epoch + 1)
                    self.Residuals_TrainingLosses = np.append(
                        self.Residuals_TrainingLosses, residuals_totalloss)
                    self.Residuals_TestLosses = np.append(
                        self.Residuals_TestLosses, residuals_testloss)
                    self._save_losses()
            
            if 14100 - (time.time() - timer) < 0:
                break
                

# ----------------------------------------------------------------------------
def main(total_epochs: int, save_every: int, batch_size: int):
    setup()
    space_dim = 10
    snapshot_path = "./plpm/models/model/MainModel10DFinal.pt"
    loss_path = "./data/loss/losses10DFinal"
    print(f"Training the model with {space_dim} input parameters")
    model, optimizer, scheduler = get_training_components(space_dim)
    train_cp = PINN_CP(num_timepoints=401,
                       space_dimension=space_dim,
                       num_cases=100000)
    nt = train_cp.normalized_timepoints()
    sCASEs_TrainDS = train_cp.build_sCASEsDS(seed=0)
    
    sCASEs_TestDS = PINN_CP(num_timepoints=401,
                            space_dimension=space_dim,
                            num_cases=100000).build_sCASEsDS(seed=1)
    
    Train_DL = prepare_dataloader(sCASEs_TrainDS, batch_size)
    Test_DL = prepare_dataloader(sCASEs_TestDS, batch_size)

    # nt_NumericalData, NumericalData_DS = train_cp.build_NumericalDataDS(
    #     data_file='numericaldata_4D_1000', step=4)
    # NumericalData_DL = prepare_dataloader(NumericalData_DS, batch_size=int(batch_size*0.1))
    
    ODEs = GoverningEqns(d=space_dim, sv_range=torch.tensor([-1, 1]))

    trainer = Trainer(model=model,
                      eqns=ODEs,
                      nt=nt,
                      Train_DataLoader=Train_DL,
                      Test_DataLoader=Test_DL,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      save_every=save_every,
                      snapshot_path=snapshot_path,
                      loss_path=loss_path,
                      NumData_flag=False)
    
    global_start = time.time()
    trainer.train(total_epochs)
    dist.destroy_process_group()
    print('Total Training Time= ', time.time() - global_start)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Lumped Parameter Model Training on Multiple GPUs')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')

    args = parser.parse_args()
    print('First model')
    main(args.total_epochs, args.save_every, args.batch_size)


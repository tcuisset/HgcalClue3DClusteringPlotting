from typing import Any, Type, Callable, Iterable, Union, Optional
import os
import matplotlib

import torch
from torch import optim, nn, utils, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import lightning.pytorch as pl
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
from torch_geometric.data import Data, InMemoryDataset, Batch
from torch_geometric.loader import DataLoader

from hists.parameters import beamEnergies
from ntupleReaders.clue_ntuple_reader import ClueNtupleReader
from ntupleReaders.computation import BaseComputation, ComputationToolBase, computeAllFromTree, NumpyArrayFilter

from ml.regression.drn.dataset_making import LayerClustersTensorMaker, RechitsTensorMaker
from ml.dynamic_reduction_network import DynamicReductionNetwork
from ml.cyclic_lr import CyclicLRWithRestarts

# beamEnergies = [20, 30, 50, 80, 100, 120, 150, 200, 250, 300]
beamEnergiesForTestSet = [30, 100, 250]

class DRNDataset(InMemoryDataset):
    def __init__(self, reader:ClueNtupleReader, datasetComputationClass:Type[BaseComputation], datasetType:str, simulation:bool=True, transform=None, pre_transform=None, pre_filter=None):
        """ Parameters : 
         - datasetComputationClass : the class, inheriting from BaseComputation, used to build the dataset form CLUE_clusters.root file.
            Must be a class that takes as __init__ args beamEnergiesToSelect, tensorFileName
            and which computes the attribute geometric_data_objects (list of PYG Data objects)
         - datasetType : can be 
              "full" : full dataset, no selection 
              "train_val" : some beam energies removed for training
              "test" : complementary of "train"
         - simulation: if True (default), will load gun energy
        """
        self.reader = reader
        self.datasetType = datasetType
        self.datasetComputationClass = datasetComputationClass
        self.simulation = simulation
        try:
            transform_name = pre_transform.__name__
        except:
            transform_name = "no_transform"
        
        if self.datasetType == "test":
            self.beamEnergiesToSelect = beamEnergiesForTestSet
        elif self.datasetType == "train_val":
            self.beamEnergiesToSelect = list(set(beamEnergies).difference(beamEnergiesForTestSet))
        elif self.datasetType == "full":
            self.beamEnergiesToSelect = beamEnergies
        else:
            raise ValueError(f"Wrong datasetType : {self.datasetType}")
        
        super().__init__(os.path.join(reader.pathToMLDatasetsFolder, "DRN", datasetComputationClass.shortName, transform_name, self.datasetType), 
                transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self):
        # Read data into huge `Data` list.
        tracksterPropComp = self.datasetComputationClass(beamEnergiesToSelect=self.beamEnergiesToSelect, 
            tensorFileName=self.processed_file_names[0], eventFilter=NumpyArrayFilter(self.reader.loadFilterArray()),
            simulation=self.simulation)
        computeAllFromTree(self.reader.tree, [tracksterPropComp], tqdm_options=dict(desc=f"Processing {self.datasetType} set"))
        data_list = tracksterPropComp.geometric_data_objects

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        # add tracksterEnergy for each event
        def tracksterEnergyTransform(data:Data) -> Data:
            data.tracksterEnergy = torch.sum(data.x[:, 3])
            return data
        data_list = [tracksterEnergyTransform(data) for data in data_list]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @property
    def processed_file_names(self):
        return "processed_data.pt"


# def ratioTransform(data:Data) -> Data:
#     data.y = torch.sum(data.x[:, 3]) / data.trueBeamEnergy
#     return data


class DRNDataModule(pl.LightningDataModule):
    def __init__(self, reader:ClueNtupleReader, datasetComputationClass:Type[BaseComputation], transformFct:Callable[[Data], Data]=None,
            multiprocess_loader:bool=True, batch_size:int=2**14, keepOnGpu:str|bool=False):
        """
        Parameters : 
         - keepOnGpu : if a devie (torch.Device or string), the dataset will be moved once to the GPU given as a device and stay there.
               If False, use the usual way of transferring each batch from CPU to GPU
        """
        super().__init__()
        self.reader = reader
        self.datasetComputationClass = datasetComputationClass
        self.transformFct = transformFct
        self.multiprocess_loader = multiprocess_loader

        self.batch_size = batch_size
        self.val_batch_size = self.batch_size
        self.test_batch_size = self.batch_size
        self.multiprocess_workers = 2
        self.keepOnGpu = keepOnGpu
        if keepOnGpu is not False and multiprocess_loader:
            raise ValueError("keepOnGpu is not compatible with multiprocess_loader. Set multiprocess_loader=False")
    
    def prepare_data(self) -> None:
        kwargs = dict(reader=self.reader, datasetComputationClass=self.datasetComputationClass, pre_transform=self.transformFct)
        self.dataset_train_val = DRNDataset(datasetType="train_val", **kwargs).shuffle()
        self.dataset_test = DRNDataset(datasetType="test", **kwargs)

    def setup(self, stage: str):
        totalev = len(self.dataset_train_val)
        self.ntrain = int(0.8*totalev)
        self.ntest = totalev - self.ntrain
        if self.keepOnGpu is not False:
            self.dataset_train_val._data.to(self.keepOnGpu)
            self.dataset_test._data.to(self.keepOnGpu)
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx: int) -> Any:
        if self.keepOnGpu is False:
            # default behaviour
            return super().transfer_batch_to_device(batch, device, dataloader_idx)
        else:
            assert torch.device(self.keepOnGpu) == torch.device(device), "Mismatch of GPUs"
            return batch # don't do anythingn the data is already on gpu

    def train_dataloader(self):
        return DataLoader(self.dataset_train_val[:self.ntrain], batch_size=self.batch_size, num_workers=self.multiprocess_workers if self.multiprocess_loader else 0)

    def val_dataloader(self):
        return DataLoader(self.dataset_train_val[self.ntrain:], batch_size=self.val_batch_size, num_workers=self.multiprocess_workers if self.multiprocess_loader else 0)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.test_batch_size, num_workers=self.multiprocess_workers if self.multiprocess_loader else 0)

    def full_dataloader(self):
        return DataLoader(torch.utils.data.ConcatDataset([self.dataset_train_val, self.dataset_test]), batch_size=self.batch_size, num_workers=self.multiprocess_workers if self.multiprocess_loader else 0)
    
    def predict_dataloader(self):
        return self.full_dataloader()

    # def predict_dataloader(self):
    #     return DataLoader(self.dataset, batch_size=1000, num_workers=4)


class BaseLossParameters:
    def mapNetworkOutputToEnergyEstimate(self, network_output_batch:torch.Tensor, data_batch:Batch):
        """ Given the network output and the corresponding input batch, compute the energy estimate by the network (energy estimate of the incident particle in GeV)
        """
        pass

    def simpleCorrectedEnergyEstimate(self, data_batch:Batch):
        """ Return the baseline prediction of energy of incident particle, without using the network.
        Usually equivalent to setting the network output to all ones
        Meant as a baseline comparison.
        The default is just the trackster energy, but subclasses can override it if they use pre-corrections
        """
        return data_batch.tracksterEnergy
    
    def rawEnergyEstimate(self, data_batch:Batch):
        """ Returns input trackster energy """
        return data_batch.tracksterEnergy

    def loss(self, network_output_batch:torch.Tensor, data_batch:Batch):
        pass

class  SimpleRelativeMSE(BaseLossParameters):
    def mapNetworkOutputToEnergyEstimate(self, network_output_batch:Batch, data_batch:Batch):
        return network_output_batch
    
    def loss(self, network_output_batch:torch.Tensor, data_batch:Batch):
        nn.functional.mse_loss(network_output_batch/data_batch.trueBeamEnergy, torch.ones_like(network_output_batch)) # Loss is MSE of E_estimate / E_beam wrt to 1

class  RatioRelativeMSE(BaseLossParameters):
    def mapNetworkOutputToEnergyEstimate(self, network_output_batch:Batch, data_batch:Batch):
        return data_batch.tracksterEnergy / network_output_batch  # probably not genenerlizable to data
    
    def loss(self, network_output_batch:torch.Tensor, data_batch:Batch):
        return nn.functional.mse_loss(network_output_batch, data_batch.tracksterEnergy / data_batch.trueBeamEnergy)

class RatioRelativeExpLoss(RatioRelativeMSE):
    def loss(self, network_output_batch:torch.Tensor, data_batch:Batch):
        return torch.mean(torch.exp(torch.abs(network_output_batch - data_batch.tracksterEnergy / data_batch.trueBeamEnergy)))

class RatioCorrectedLoss(BaseLossParameters):
    """ Default for coefs : [-0.2597882 , -0.24326517,  1.01537901] """
    def __init__(self, coefs:list[float]) -> None:
        super().__init__()
        self.a = torch.tensor(coefs)
    
    def _correctTracksterEnergy(self, data_batch: Batch) -> torch.Tensor:
        """ Corrects the trackster energy so the mean is a non-biased estimator of incident particle energy """
        rawTracksterEnergy = data_batch.tracksterEnergy
        return rawTracksterEnergy * 1/ (self.a[0] * rawTracksterEnergy**(self.a[1]) + self.a[2] )
    
    def mapNetworkOutputToEnergyEstimate(self, network_output_batch:torch.Tensor, data_batch: Batch) -> torch.Tensor:
        return network_output_batch * self._correctTracksterEnergy(data_batch)

    def simpleCorrectedEnergyEstimate(self, data_batch: Batch):
        return self._correctTracksterEnergy(data_batch)
    
    def loss(self, network_output_batch:torch.Tensor, data_batch:Batch):
        return nn.functional.mse_loss(self.mapNetworkOutputToEnergyEstimate(network_output_batch, data_batch), data_batch.trueBeamEnergy)

    @property
    def hyperparameters(self) -> dict[str, float]:
        return {"tracksterEnergyCorrection_0_prop" : self.a[0],
                "tracksterEnergyCorrection_1_exp" : self.a[1],
                "tracksterEnergyCorrection_2_cst" : self.a[2]}


class LRSchedulerAdapter:
    def instantiate(self, optimizer, batch_size, epoch_size) -> LRScheduler:
        pass

class CyclicLRWithRestartsAdapter(LRSchedulerAdapter):
    def __init__(self, restart_period:int=100, t_mult:float=2, verbose:bool=False,
                 policy:str="cosine", policy_fn=None, min_lr:float=1e-7,
                 gamma:float=1.0, triangular_step:float=0.5) -> None:
        self.kwargs_stored = {"restart_period":restart_period, "t_mult":t_mult, "verbose":verbose,
            "policy":policy, "policy_fn":policy_fn, "min_lr":min_lr,
            "gamma":gamma, "triangular_step":triangular_step}

    def instantiate(self, optimizer, batch_size, epoch_size) -> LRScheduler:
        return CyclicLRWithRestarts(optimizer, batch_size=batch_size, epoch_size=epoch_size, **self.kwargs_stored)


class DRNModule(pl.LightningModule):
    #defaultOptimizer = lambda 
    def __init__(self, drn:nn.Module, loss:BaseLossParameters,
        optimizer:Callable[[Iterable], Optimizer],
        lr_scheduler:Optional[Union[Callable[[Optimizer], LRScheduler], CyclicLRWithRestartsAdapter]]=None,
        #optimizer:str="AdamW", optimizer_params:dict=dict(lr=1e-3, weight_decay=1e-3),
        #        scheduler:str="CyclicLRWithRestarts", scheduler_params:dict=dict(restart_period=80, t_mult=1.2, policy="cosine")
            ):
        """ Parameters :
         - drn : the module
         - scheduler : can be "CyclicLRWithRestarts", "default"
         - loss : can be "mse_relative", "mse_ratio" 
        """
        super().__init__()
        self.drn = drn
        self.loss_params = loss

        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        hp = dict()
        try:
            hp.update(drn.hyperparameters)
        except:
            pass
        self.hyperparameters = hp


    def _common_prediction_step(self, batch):
        result = self(batch)
        loss = self.loss_params.loss(result, batch)
        return result, loss

    def training_step(self, batch, batch_idx):
        result, loss = self._common_prediction_step(batch)
         # NB: setting batch_size is important as otherwise Lightning does not know how to compute the batch size from PYG DataBatch batches
        self.log("Loss/Training", loss, batch_size=batch.num_graphs, on_step=True, on_epoch=True)
        return {"loss":loss, "output":result}

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        if self.scheduler is None:
            return optimizer
        try:
            scheduler = self.scheduler(optimizer)
        except:
            # need to load dataloader to access data size and batch_size.
            #  Trick taken from https://github.com/Lightning-AI/lightning/issues/10430#issuecomment-1487753339
            self.trainer.fit_loop.setup_data() 
            scheduler = self.scheduler.instantiate(optimizer, 
                self.trainer.train_dataloader.batch_size, 
                len(self.trainer.train_dataloader.dataset))
        
        if isinstance(scheduler, CyclicLRWithRestarts):
            scheduler_config = {
                "scheduler" : scheduler,
                "interval" : "step",
            }
        else:
            scheduler_config = scheduler
        # if self.optimizer_name == "AdamW":
        #     optimizer = optim.AdamW(self.parameters(), **self.optimizer_params)
        # else:
        #     raise ValueError("Optimizer name")
        # if self.scheduler_name is None:
        #     return optimizer
        # elif self.scheduler_name == "CyclicLRWithRestarts":
        #     # need to load dataloader to access data size and batch_size.
        #     # Trick taken from https://github.com/Lightning-AI/lightning/issues/10430#issuecomment-1487753339
        #     self.trainer.fit_loop.setup_data() 

        #     scheduler = CyclicLRWithRestarts(optimizer, self.trainer.train_dataloader.batch_size, 
        #         len(self.trainer.train_dataloader.dataset), **self.scheduler_params)
        #     scheduler_config = {
        #         "scheduler" : scheduler,
        #         "interval" : "step",
        #     }
                
        # else:
        #     raise ValueError("DRNModule.scheduler")
        return {
            "optimizer" : optimizer,
            "lr_scheduler" : scheduler_config
        }

    def lr_scheduler_step(self, scheduler, metric):
        if isinstance(scheduler, CyclicLRWithRestarts):
            scheduler.batch_step()
        else:
            super().lr_scheduler_step(scheduler, metric)
    
    def on_train_epoch_start(self) -> None:
        scheduler = self.lr_schedulers()
        if isinstance(scheduler, CyclicLRWithRestarts):
            scheduler.step()
    
    def forward(self, data):
        return self.drn(data)
    

    def validation_step(self, batch, batch_idx):
        result, loss = self._common_prediction_step(batch)
        self.log("Loss/Validation", loss, batch_size=batch.num_graphs, on_step=True, on_epoch=True)

        return {"loss":loss, "output":result}
    
    def test_step(self, batch, batch_idx):
        result, loss = self._common_prediction_step(batch)
        self.log("Loss/Test", loss, batch_size=batch.num_graphs, on_step=True, on_epoch=True)
        return {"loss":loss, "output":result}

    def on_fit_start(self):
        try:
            matplotlib.use("Agg")
            # Overlay plots. See https://stackoverflow.com/a/71524389
            tb = self.logger.experiment
            tb.add_custom_scalars({
                "Loss" : {
                    "Epoch" : ["Multiline", ["Loss/Training_epoch", "Loss/Validation_epoch"]],
                    "Step" : ["Multiline", ["Loss/Training_step", "Loss/Validation_step"]],
                }
            })
        except AttributeError:
            pass



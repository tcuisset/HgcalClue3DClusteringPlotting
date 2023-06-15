from typing import Type, Callable
import os
import matplotlib

import torch
from torch import optim, nn, utils, Tensor
from torch.utils.tensorboard import SummaryWriter
import lightning.pytorch as pl
from torch_geometric.data import Data, InMemoryDataset, Batch
from torch_geometric.loader import DataLoader

from hists.parameters import beamEnergies
from ntupleReaders.clue_ntuple_reader import ClueNtupleReader
from ntupleReaders.computation import BaseComputation, ComputationToolBase, computeAllFromTree, NumpyArrayFilter

from ml.regression.drn.dataset_making import LayerClustersTensorMaker, RechitsTensorMaker
from ml.dynamic_reduction_network import DynamicReductionNetwork
from ml.cyclic_lr import CyclicLRWithRestarts
from ml.regression.drn.plot import scatterPredictionVsTruth

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
        super().__init__(os.path.join(reader.pathToMLDatasetsFolder, "DRN", datasetComputationClass.shortName, transform_name, self.datasetType), 
                transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self):
        # Read data into huge `Data` list.
        if self.datasetType == "test":
            beamEnergiesToSelect = beamEnergiesForTestSet
        elif self.datasetType == "train_val":
            beamEnergiesToSelect = list(set(beamEnergies).difference(beamEnergiesForTestSet))
        elif self.datasetType == "full":
            beamEnergiesToSelect = beamEnergies
        else:
            raise ValueError(f"Wrong datasetType : {self.datasetType}")
        
        tracksterPropComp = self.datasetComputationClass(beamEnergiesToSelect=beamEnergiesToSelect, 
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
            multiprocess_loader:bool=True, batch_size:int=2**14):
        super().__init__()
        self.reader = reader
        self.datasetComputationClass = datasetComputationClass
        self.transformFct = transformFct
        self.multiprocess_loader = multiprocess_loader

        self.batch_size = batch_size
        self.val_batch_size = self.batch_size
        self.test_batch_size = self.batch_size
        self.multiprocess_workers = 2
    
    def prepare_data(self) -> None:
        kwargs = dict(reader=self.reader, datasetComputationClass=self.datasetComputationClass, pre_transform=self.transformFct)
        self.dataset_train_val = DRNDataset(datasetType="train_val", **kwargs).shuffle()
        self.dataset_test = DRNDataset(datasetType="test", **kwargs)

    def setup(self, stage: str):
        totalev = len(self.dataset_train_val)
        self.ntrain = int(0.8*totalev)
        self.ntest = totalev - self.ntrain

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
        pass
    
    def loss(self, network_output_batch:torch.Tensor, data_batch:Batch):
        pass

class  SimpleRelativeMSE(nn.Module, BaseLossParameters):
    def mapNetworkOutputToEnergyEstimate(self, network_output_batch:Batch, data_batch:Batch):
        return network_output_batch
    
    def loss(self, network_output_batch:torch.Tensor, data_batch:Batch):
        nn.functional.mse_loss(network_output_batch/data_batch.trueBeamEnergy, torch.ones_like(network_output_batch)) # Loss is MSE of E_estimate / E_beam wrt to 1

class  RatioRelativeMSE(nn.Module, BaseLossParameters):
    def mapNetworkOutputToEnergyEstimate(self, network_output_batch:Batch, data_batch:Batch):
        return network_output_batch * data_batch.trueBeamEnergy
    
    def loss(self, network_output_batch:torch.Tensor, data_batch:Batch):
        nn.functional.mse_loss(network_output_batch, data_batch.tracksterEnergy / data_batch.trueBeamEnergy)


class RatioCorrectedLoss(nn.Module, BaseLossParameters):
    """ Default for coefs : [-0.2597882 , -0.24326517,  1.01537901] """
    def __init__(self, coefs:list[float]) -> None:
        super().__init__()
        self.a = torch.tensor(coefs)
    
    def mapNetworkOutputToEnergyEstimate(self, network_output_batch:torch.Tensor, data_batch: Batch) -> torch.Tensor:
        rawTracksterEnergy = data_batch.tracksterEnergy
        correctedTracksterEnergy = rawTracksterEnergy * 1/ (self.a[0] * rawTracksterEnergy**(self.a[1]) + self.a[2] )
        return network_output_batch * correctedTracksterEnergy

    def loss(self, network_output_batch:torch.Tensor, data_batch:Batch):
        return nn.functional.mse_loss(self.mapNetworkOutputToEnergyEstimate(network_output_batch, data_batch), data_batch.trueBeamEnergy)


class DRNModule(pl.LightningModule):
    def __init__(self, drn:nn.Module, loss:BaseLossParameters, scheduler:str="CyclicLRWithRestarts"):
        """ Parameters :
         - drn : the module
         - scheduler : can be "CyclicLRWithRestarts", "default"
         - loss : can be "mse_relative", "mse_ratio" 
        """
        super().__init__()
        self.drn = drn
        self.validation_predictions = []
        self.validation_trueBeamEnergy = []
        self.scheduler_name = scheduler
        self.loss_params = loss

    def _common_prediction_step(self, batch):
        result = self(batch)
        loss = self.loss_params.loss(result, batch)
        return result, loss

    def training_step(self, batch, batch_idx):
        result, loss = self._common_prediction_step(batch)
         # NB: setting batch_size is important as otherwise Lightning does not know how to compute the batch size from PYG DataBatch batches
        self.log("Loss/Training", loss, batch_size=batch.num_graphs, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-3)
        if self.scheduler_name == "CyclicLRWithRestarts":
            # need to load dataloader to access data size and batch_size.
            # Trick taken from https://github.com/Lightning-AI/lightning/issues/10430#issuecomment-1487753339
            self.trainer.fit_loop.setup_data() 

            scheduler = CyclicLRWithRestarts(optimizer, self.trainer.train_dataloader.batch_size, 
                len(self.trainer.train_dataloader.dataset), restart_period=80, t_mult=1.2, policy="cosine")
            return {
                "optimizer" : optimizer,
                "lr_scheduler" : {
                    "scheduler" : scheduler,
                    "interval" : "step",
                }
            }
        elif self.scheduler_name == "default":
            return optimizer
        else:
            raise ValueError("DRNModule.scheduler")

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

        self.validation_predictions.append(result)
        self.validation_trueBeamEnergy.append(batch.trueBeamEnergy)

        return result
    
    def test_step(self, batch, batch_idx):
        result, loss = self._common_prediction_step(batch)
        self.log("Loss/Test", loss, batch_size=batch.num_graphs, on_step=True, on_epoch=True)

        self.validation_predictions.append(result)
        self.validation_trueBeamEnergy.append(batch.trueBeamEnergy)

    def _test_val_common_epoch_end(self):
        validation_pred = torch.cat([x.detach() for x in self.validation_predictions])
        self.validation_predictions.clear()
        validation_trueBeamEnergy = torch.cat([x.detach() for x in self.validation_trueBeamEnergy])
        self.validation_trueBeamEnergy.clear()

        try:
            tbWriter:SummaryWriter = self.logger.experiment
            tbWriter.add_histogram("Validation/Prediction",
                validation_pred, self.current_epoch)
            
            # tbWriter.add_figure(
            #     "Validation/Scatter",
            #     scatterPredictionVsTruth(validation_trueBeamEnergy.cpu().numpy(), 
            #         self.loss_params.mapNetworkOutputToEnergyEstimate(validation_pred))
            # )

            # if self.prediction_type == "absolute":
            #     self.logger.experiment.add_histogram("Validation/Pred-truth / truth", 
            #         (validation_pred-validation_trueBeamEnergy)/validation_trueBeamEnergy, self.current_epoch)

            #     self.logger.experiment.add_figure("Validation/Scatter",
            #         scatterPredictionVsTruth(validation_trueBeamEnergy.cpu().numpy(), validation_pred.cpu().numpy(), epoch=self.current_epoch),
            #         self.current_epoch)
            # elif self.prediction_type == "ratio":
            #     self.logger.experiment.add_histogram("Validation/Prediction histogram", 
            #         validation_pred, self.current_epoch)
                
            #     self.logger.experiment.add_figure("Validation/Scatter",
            #         scatterPredictionVsTruth(validation_trueBeamEnergy.cpu().numpy(), (validation_pred*validation_trueBeamEnergy).cpu().numpy(), epoch=self.current_epoch),
            #         self.current_epoch)
                
        except AttributeError:
            pass # no logger
    
    def on_validation_epoch_end(self) -> None:
        self._test_val_common_epoch_end()
    def on_test_epoch_end(self) -> None:
        self._test_val_common_epoch_end()


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



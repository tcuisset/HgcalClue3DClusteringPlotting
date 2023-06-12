


from typing import Any, Optional
import os

import torch
from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from hists.parameters import beamEnergies
from ntupleReaders.clue_ntuple_reader import ClueNtupleReader
from ntupleReaders.computation import BaseComputation, ComputationToolBase, computeAllFromTree, NumpyArrayFilter
from ml.regression.rechits import RechitsTensorMaker

from ml.dynamic_reduction_network import DynamicReductionNetwork
from ml.cyclic_lr import CyclicLRWithRestarts
from ml.regression.drn.plot import scatterPredictionVsTruth

# beamEnergies = [20, 30, 50, 80, 100, 120, 150, 200, 250, 300]
beamEnergiesForTestSet = [30, 100, 250]

class RechitsDataset(InMemoryDataset):
    def __init__(self, reader:ClueNtupleReader, is_test_set=False, transform=None, pre_transform=None, pre_filter=None):
        self.reader = reader
        self.is_test_set = is_test_set
        super().__init__(os.path.join(reader.pathToMLDatasetsFolder, "rechits_DRN"), transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        if self.is_test_set:
            return ['rechits_DRN_processed_test.pt']
        else:
            return ['rechits_DRN_processed.pt']

    def process(self):
        # Read data into huge `Data` list.
        if self.is_test_set:
            beamEnergiesToSelect = beamEnergiesForTestSet
        else:
            beamEnergiesToSelect = list(set(beamEnergies).difference(beamEnergiesForTestSet))
        
        tracksterPropComp = RechitsTensorMaker(beamEnergiesToSelect=beamEnergiesToSelect, 
            tensorFileName=self.processed_file_names[0], eventFilter=NumpyArrayFilter(self.reader.loadFilterArray()))
        computeAllFromTree(self.reader.tree, [tracksterPropComp], tqdm_options=dict(desc="Processing rechits data" + (" (test set)" if self.is_test_set else "")))
        data_list = tracksterPropComp.geometric_data_objects

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



# beamEnergies = [20, 30, 50, 80, 100, 120, 150, 200, 250, 300]
defaultBeamEnergiesForTest = [30, 120, 250]

class RechitsDRNDataModule(pl.LightningDataModule):
    def __init__(self, reader:ClueNtupleReader=None):
        super().__init__()
        self.reader = reader
    
    def setup(self, stage: str):
        self.dataset_train_val = RechitsDataset(self.reader, is_test_set=False).shuffle()
        totalev = len(self.dataset_train_val)
        self.ntrain = int(0.8*totalev)
        self.ntest = totalev - self.ntrain

        self.train_batch_size = 200
        self.val_batch_size = 100

        self.dataset_test = RechitsDataset(self.reader, is_test_set=True)
        self.test_batch_size = 400

    def train_dataloader(self):
        return DataLoader(self.dataset_train_val[:self.ntrain], batch_size=self.train_batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.dataset_train_val[self.ntrain:], batch_size=self.val_batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.test_batch_size, num_workers=2)

    # def predict_dataloader(self):
    #     return DataLoader(self.dataset, batch_size=1000, num_workers=4)


class RechitsDRNModule(pl.LightningModule):
    def __init__(self, drn:nn.Module=None, scheduler:str="CyclicLRWithRestarts"):
        super().__init__()
        self.drn = drn
        self.validation_predictions = []
        self.validation_trueBeamEnergy = []
        self.scheduler_name = scheduler

    def _common_prediction_step(self, batch):
        result = self(batch)
        loss = nn.functional.mse_loss(result/batch.trueBeamEnergy, torch.ones_like(result)) # Loss is MSE of E_estimate / E_beam wrt to 1
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
            raise ValueError("RechitsDRNModule.scheduler")

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

    def on_validation_epoch_end(self) -> None:
        self._test_val_common_epoch_end()

    def _test_val_common_epoch_end(self):
        validation_pred = torch.cat([x.detach() for x in self.validation_predictions])
        self.validation_predictions.clear()
        validation_trueBeamEnergy = torch.cat([x.detach() for x in self.validation_trueBeamEnergy])
        self.validation_trueBeamEnergy.clear()

        try:
            self.logger.experiment.add_histogram("Validation/Pred-truth / truth", 
                (validation_pred-validation_trueBeamEnergy)/validation_trueBeamEnergy, self.current_epoch)

            self.logger.experiment.add_figure("Validation/Scatter",
                scatterPredictionVsTruth(validation_trueBeamEnergy.cpu().numpy(), validation_pred.cpu().numpy(), epoch=self.current_epoch),
                self.current_epoch)
        except AttributeError:
            pass # no logger
    
    def on_validation_epoch_end(self) -> None:
        self._test_val_common_epoch_end()
    def on_test_epoch_end(self) -> None:
        self._test_val_common_epoch_end()








import os

import numpy as np
import hist
import typing
import pandas as pd

import torch
from torch import optim, nn, utils, Tensor
from torch.utils.data import DataLoader, TensorDataset
import lightning.pytorch as pl

from hists.dataframe import DataframeComputations
from hists.custom_hists import beamEnergiesAxis, layerAxis_custom

from ntupleReaders.clue_ntuple_reader import ClueNtupleReader
from ntupleReaders.computation import BaseComputation, ComputationToolBase
from ntupleReaders.tools import DataframeComputationsToolMaker
from energy_resolution.sigma_over_e import SigmaOverEComputations, fitSigmaOverE, plotSCAsEllipse, SigmaOverEPlotElement

from ml.dataset_maker.tensors import SimpleTensorMakingComputation



neededBranchesGlobal = ["clus3D_energy", "clus3D_idxs", "clus3D_size",  'clus3D_x', 'clus3D_y', 'clus3D_z',
            "rechits_energy", "rechits_layer", "beamEnergy", "trueBeamEnergy", 'clus2D_x', 'clus2D_y', 'clus2D_z', 'clus2D_energy', 'clus2D_layer', 'clus2D_size', 'clus2D_rho', 'clus2D_delta', 'clus2D_pointType']

def makeDataframeFromComp(comp:DataframeComputations) -> pd.DataFrame:
    df = comp.clusters3D_intervalHoldingFractionOfEnergy_joined(fraction=0.7)
    df = df[df.index.isin(comp.clusters3D_largestClusterIndex_fast)]
    return df
    
def makeInputTensorFromDf(df:pd.DataFrame):
    # converting to float32 is necessary (otherwise everything gets converted to double)
    return torch.tensor(df[["clus3D_energy", "intervalFractionEnergy_length", "intervalFractionEnergy_minLayer", "intervalFractionEnergy_maxLayer"]].to_numpy(dtype=np.float32))

class TracksterPropertiesTensorMaker(SimpleTensorMakingComputation):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs, neededBranches=neededBranchesGlobal,
            rechits_columns=["rechits_energy", "rechits_layer"], tensorFileName="tracksterProperties")

    def computeTensor(self, comp:DataframeComputations) -> torch.Tensor:
        df = makeDataframeFromComp(comp).join(comp.trueBeamEnergy.trueBeamEnergy)
        
        return makeInputTensorFromDf(df), torch.tensor(df.trueBeamEnergy.to_numpy(dtype=np.float32)), torch.tensor(df.beamEnergy.to_numpy(dtype=np.float32))


def makeHist():
    return hist.Hist(beamEnergiesAxis(), 
        hist.axis.Regular(bins=2000, start=0, stop=350, name="clus3D_mainTrackster_energy_pred", label="Main trackster energy, predicetd by NN (GeV)"))

class TracksterEnergyFromNetwork(BaseComputation):
    def __init__(self, model, **kwargs) -> None:
        super().__init__(**kwargs, neededBranches=neededBranchesGlobal,
            neededComputationTools=[DataframeComputationsToolMaker(rechits_columns=["rechits_layer", "rechits_energy"])]
        )
        self.model = model
        self.h = makeHist()
    
    def processBatch(self, array, report, computationTools:dict[typing.Type[ComputationToolBase], ComputationToolBase]) -> None:
        comp = computationTools[DataframeComputations]
        df = makeDataframeFromComp(comp)
        prediction = torch.squeeze(self.model(makeInputTensorFromDf(df)))
        self.h.fill(df.beamEnergy, prediction.detach().numpy())



def fillHistogramsFromTensorBatches(predictions:list[torch.Tensor], full_dataloader:torch.utils.data.DataLoader):
    h = makeHist()

    for prediction_batch, input_batch in zip(predictions, full_dataloader):
        h.fill(input_batch[2], prediction_batch)
    # model.eval()
    # model.cpu()
    # with torch.no_grad():
    #     # .tensors[2] is nominal beam energy
    #     h.fill(tensorDataset.tensors[2].detach().cpu().numpy(), torch.squeeze(model(tensorDataset.tensors[0].detach())).numpy())
    return h


class TracksterPropDataModule(pl.LightningDataModule):
    def __init__(self, reader:ClueNtupleReader=None):
        if reader is None:
            reader = ClueNtupleReader("v40", "cmssw", "sim_proton_v46_patchMIP")
        super().__init__()
        self.reader = reader
    
    def setup(self, stage: str):
        self.dataset = TensorDataset(*torch.load(os.path.join(self.reader.pathToFolder, "tracksterProperties.pt")))
        totalev = len(self.dataset)
        ntrain = int(0.8*totalev)
        ntest = totalev - ntrain

        self.train_batch_size = 200
        self.test_batch_size = 100
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(self.dataset, [ntrain, ntest])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=4)

    # def test_dataloader(self):
    #     return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=1000, num_workers=4)



class BasicHiddenLayerModel(torch.nn.Module):
    def __init__(self, hidden_size=10) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(4, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )
    def forward(self, x):
        return self.net(x)

class TracksterPropModule(pl.LightningModule):
    def __init__(self, net:nn.Module=None):
        super().__init__()
        if net is None:
            net = BasicHiddenLayerModel()
        self.net = net
        

    def _common_prediction_step(self, batch):
        tracksterProp, trueBeamEnergy = batch[0], batch[1]
        result = torch.squeeze(self.net(tracksterProp))
        loss = nn.functional.mse_loss(result/trueBeamEnergy, torch.ones_like(result)) # Loss is MSE of E_estimate / E_beam wrt to 1
        return result, loss

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        result, loss = self._common_prediction_step(batch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        result, loss = self._common_prediction_step(batch)
        self.log("val_loss", loss)
        return result

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def forward(self, x):
        return torch.squeeze(self.net(x[0]))



class SigmaOverECallback(pl.Callback):
    def __init__(self, every_n_epochs=100) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
    
    def _fillHistogram(self, sigma_over_e_dataloader:DataLoader, pl_module: pl.LightningModule):
        h = makeHist()
        for i, batch in enumerate(sigma_over_e_dataloader):
            tracksterProp, trueBeamEnergy, nominalBeamEnergy = batch[0], batch[1], batch[2]
            tracksterProp = pl_module.transfer_batch_to_device(tracksterProp, pl_module.device, dataloader_idx=0)
            pred = pl_module.predict_step(tracksterProp, i)
            h.fill(nominalBeamEnergy.detach().numpy(), pred.detach().cpu().numpy())
        return h
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        if trainer.sanity_checking:  # optional skip
            return
        
        print("Start predicting!")
        try:
            h = self._fillHistogram(trainer.datamodule.predict_dataloader(), pl_module)
            
            E_res_fitResult = fitSigmaOverE(SigmaOverEComputations().compute(h, tqdm_dict=dict(disable=True), multiprocess=True))
            fig = plotSCAsEllipse([SigmaOverEPlotElement(legend="From ML", fitResult=E_res_fitResult)])
            pl_module.logger.experiment.add_figure("Validation/SigmaOverEEllipse", fig, trainer.current_epoch)
        except:
            print("SigmaOverE fit failed")
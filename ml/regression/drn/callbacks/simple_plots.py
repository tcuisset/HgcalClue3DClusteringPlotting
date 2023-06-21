import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import lightning.pytorch as pl

import hists.parameters

def scatterPredictionVsTruth(true_values:np.ndarray, pred_values:np.ndarray, epoch=None) -> mpl.figure.Figure:
    fig, ax = plt.subplots(figsize=(20, 20), dpi=50)
    ax.scatter(true_values, pred_values)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes) # oblique line

    ax.set_xlim(left=0)
    ax.set_xlabel("True beam energy (GeV)")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Predicted beam energy (GeV)")

    hep.cms.text("Preliminary Simulation", ax=ax)
    if epoch is not None:
        hep.cms.lumitext(f"Epoch : {epoch}", ax=ax)
    return fig

# def scatterNetworkOutput(trueBeamEnergies:np.ndarray, network_outputs:np.ndarray, epoch=None) -> mpl.figure.Figure:
#     fig, ax = plt.subplots(figsize=(20, 20), dpi=50)
#     ax.scatter(trueBeamEnergies, network_outputs)
#     ax.plot([0, 1], [0, 1], transform=ax.transAxes) # oblique line

#     ax.set_xlim(left=0)
#     ax.set_xlabel("True beam energy (GeV)")
#     ax.set_ylim(bottom=0)
#     ax.set_ylabel("Network output")

#     hep.cms.text("Preliminary Simulation", ax=ax)
#     if epoch is not None:
#         hep.cms.lumitext(f"Epoch : {epoch}", ax=ax)
#     return fig

def violin(beamEnergies:np.ndarray, yValues:np.ndarray, epoch=None) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    fig, ax = plt.subplots(figsize=(15, 15), dpi=50)

    yValues_list = []
    xValues = []
    """ List of y values for each beam energy """
    for beamEnergy in hists.parameters.beamEnergies:
        cur_yValues = yValues[beamEnergies == beamEnergy]
        if len(cur_yValues) > 0:
            yValues_list.append(cur_yValues)
            xValues.append(beamEnergy)
    
    ax.violinplot(yValues_list)
    ax.set_xlabel("Beam energy (GeV)")
    ax.set_xticks(np.arange(1, len(xValues) + 1), labels=xValues)
    hep.cms.text("Preliminary Simulation", ax=ax)
    if epoch is not None:
        hep.cms.lumitext(f"Epoch : {epoch}", ax=ax)

    return fig, ax

def moveListOfTensorsToNumpy(tensors:list[torch.Tensor]) -> np.ndarray:
    return np.concatenate([tensor.detach().cpu().numpy() for tensor in tensors])

class SimplePlotsCallback(pl.Callback):
    def __init__(self) -> None:
        self._clearLists()
    
    def _clearLists(self):
        self.tracksterEnergyEstimates = []
        self.trueBeamEnergies = []
        self.beamEnergies = []
        self.networkOutputs = []

    def _moveAllToCpu(self):
        self.tracksterEnergyEstimates = moveListOfTensorsToNumpy(self.tracksterEnergyEstimates)
        self.trueBeamEnergies = moveListOfTensorsToNumpy(self.trueBeamEnergies)
        self.beamEnergies = moveListOfTensorsToNumpy(self.beamEnergies)
        self.networkOutputs = moveListOfTensorsToNumpy(self.networkOutputs)

    def _plot(self, pl_module: pl.LightningModule, epochName):
        """ epochName can be Validation or Training """
        tbWriter:SummaryWriter = pl_module.logger.experiment

        tbWriter.add_figure(
            f"Scatter/{epochName}",
            scatterPredictionVsTruth(self.trueBeamEnergies, self.tracksterEnergyEstimates, epoch=pl_module.current_epoch),
            pl_module.current_epoch
        )

        tbWriter.add_histogram(
            f"Prediction histograms/{epochName} prediction",
            self.networkOutputs,
            pl_module.current_epoch
        )

        pred_over_truth_array = (self.tracksterEnergyEstimates - self.trueBeamEnergies)/self.trueBeamEnergies
        tbWriter.add_histogram(
            f"Prediction histograms/{epochName} Pred-truth / truth",
            pred_over_truth_array,
            pl_module.current_epoch
        )

        for beamEnergy in hists.parameters.beamEnergies:
            try:
                tbWriter.add_histogram(
                    f"HistsPerEnergy/{epochName} prediction - {beamEnergy} GeV",
                    self.networkOutputs[self.beamEnergies == beamEnergy],
                    pl_module.current_epoch
                )

                tbWriter.add_histogram(
                    f"HistsPerEnergy/{epochName} Pred-truth / truth - {beamEnergy} GeV",
                    pred_over_truth_array[self.beamEnergies == beamEnergy],
                    pl_module.current_epoch
                )
            except ValueError: # the input has no values to histogram
                print(f"SimplePlotsCallbacks : skipping histogramming for beam energy={beamEnergy} GeV due to no values to histogram")
        
        # Violin plots
        fig, ax = violin(beamEnergies=self.beamEnergies, yValues=self.networkOutputs, epoch=pl_module.current_epoch)
        ax.set_ylabel("Network outputs")
        tbWriter.add_figure(f"Violin/{epochName} prediction", fig, pl_module.current_epoch)

        fig, ax = violin(beamEnergies=self.beamEnergies, yValues=pred_over_truth_array, epoch=pl_module.current_epoch)
        ax.set_ylabel("(Prediction - truth) / truth")
        tbWriter.add_figure(f"Violin/{epochName} Pred-truth / truth", fig, pl_module.current_epoch)


    def _common_batch_end(self, pl_module: pl.LightningModule, outputs, batch):
        self.networkOutputs.append(outputs)
        self.tracksterEnergyEstimates.append(pl_module.loss_params.mapNetworkOutputToEnergyEstimate(outputs, batch))
        self.trueBeamEnergies.append(batch.trueBeamEnergy)
        self.beamEnergies.append(batch.beamEnergy)

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        self._common_batch_end(pl_module, outputs["output"], batch)
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int) -> None:
        self._common_batch_end(pl_module, outputs["output"], batch)

    def _common_epoch_end(self, pl_module: pl.LightningModule, epochName:str):
        if len(self.networkOutputs) == 0:
            return
        self._moveAllToCpu()
        self._plot(pl_module, epochName)
        self._clearLists()
    
    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # for some reason on_train_epoch_end runs after the validation epoch. So we need to deal with training metrics before the validation starts
        self._common_epoch_end(pl_module, "Training")
    # def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
    #     self._common_epoch_end(pl_module, "Training")
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._common_epoch_end(pl_module, "Validation")

    

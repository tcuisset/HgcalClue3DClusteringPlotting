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

def scatterPredictionVsTruth(true_values:np.ndarray, pred_values:np.ndarray, original_values:np.ndarray=None, epoch=None,
                marker_size=36/2) -> mpl.figure.Figure:
    """ Scatter plot of predicted beam energy (pred_values) vs true beam energy (true_values)
    Parameters : 
     - original_values : facultative, plot raw trackster energies, in another color, to compare 
     - epoch : epoch to write on plot
    """
    fig, ax = plt.subplots(figsize=(20, 20), dpi=50)
    if original_values is not None:
        ax.scatter(true_values, original_values, s=marker_size, c="red", alpha=0.15 if len(true_values) > 1e5 else 0.5, label="Trackster energy")

    ax.scatter(true_values, pred_values, s=marker_size, alpha=0.3 if len(true_values) > 1e5 else 0.8, label="Predicted true beam energy")

    ax.plot([0, 500], [0, 500]) # oblique line

    ax.set_xlim(0, 350)
    ax.set_xlabel("True beam energy (GeV)")
    ax.set_ylim(0, 350)
    ax.set_ylabel("Predicted beam energy (GeV)")

    ax.legend()

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
    
    ax.violinplot(yValues_list, showmeans=True, quantiles=[[0.25, 0.75]]*len(xValues), showextrema=False)
    ax.set_xlabel("Beam energy (GeV)")
    ax.set_xticks(np.arange(1, len(xValues) + 1), labels=xValues)
    hep.cms.text("Preliminary Simulation", ax=ax)
    if epoch is not None:
        hep.cms.lumitext(f"Epoch : {epoch}", ax=ax)

    return fig, ax

import matplotlib.patches as mpatches

def doubleViolin(beamEnergies:np.ndarray, bare_preds:np.ndarray, network_preds:np.ndarray, epoch=None, normBy:str=None, overlapViolins=False) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """ Make two viloin plots side by side showing the distribution of trackster energy for the base and predicted by the network 
    Parameters : 
     - beamEnergies : array of nominal beam energies, one value per event
     - bare_preds : array of bare predictions for trackster energy (same length as beamEnergies)
     - network_preds : same but for network prediction of same qty
     - epoch : to plot at top right
     - normBy : can be None (plot trackster energy in GeV in y axis) or "incidentBeamEnergy" (plot tracksterEnergy/incidentBeamEnergy in y)
     - overlapViolins : if True, violins are overlaid. If False, they are plotted side-by-side
    Returns a tuple Figure, Axes  
    """
    if overlapViolins:
        figsize=(15, 15)
    else:
        figsize=(20, 10)
    fig, ax = plt.subplots(figsize=figsize, dpi=50)

    bare_listOfLists = []
    net_listOfLists = []
    xValues = []

    

    """ List of y values for each beam energy """
    for beamEnergy in hists.parameters.beamEnergies:
        cur_bare = bare_preds[beamEnergies == beamEnergy]
        cur_net = network_preds[beamEnergies == beamEnergy]

        if normBy is None:
            norm = lambda x : x
        elif normBy == "incidentBeamEnergy":
            def norm(l:np.ndarray):
                return l / hists.parameters.synchrotronBeamEnergiesMap[beamEnergy]
        else:
            raise ValueError(f"Wrong normBy : {normBy}")

        if len(cur_bare) > 0:
            xValues.append(beamEnergy)
            bare_listOfLists.append(norm(cur_bare))
            net_listOfLists.append(norm(cur_net))
    

    labels = []
    def add_violin(violin, label, color=None):
        for pc in violin['bodies']:
            if color:
                pc.set_facecolor(color)
                pc.set_edgecolor("black")
            pc.set_alpha(0.7)
        violin_dict_copy = violin.copy()
        del violin_dict_copy["bodies"]
        for pc in violin_dict_copy.values():#[violin["cmeans"], violin["cmins"], violin["cmaxes"], violin["cbars"], violin["cmedians"], violin["cquantiles"]]:
            pc.set_color("black")
        color_patch = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color_patch), label))

    violin_kwargs = dict(showmeans=True, quantiles=[[0.25, 0.75]]*len(xValues), showextrema=False)
    center_positions = np.arange(1, len(xValues) + 1)
    if overlapViolins:
        positions = center_positions, center_positions
    else:
        lat_spread = 0.2
        positions = center_positions - lat_spread, center_positions + lat_spread
        violin_kwargs["widths"] = 0.35
    ax.set_xticks(center_positions, labels=xValues)

    
    add_violin(ax.violinplot(bare_listOfLists, positions=positions[0], **violin_kwargs), "Raw prediction", color="tab:red")
    add_violin(ax.violinplot(net_listOfLists, positions=positions[1], **violin_kwargs),"Network prediction", color="tab:blue")
    ax.set_xlabel("Beam energy (GeV)")
    if normBy is None:
        ax.set_ylabel("(Predicted) trackster energy (GeV)")
    elif normBy == "incidentBeamEnergy":
        ax.set_ylabel("Ratio of (predicted) trackster energy over incident beam energy", fontsize=20)
    else:
        assert False, "normBy case is missing"
    
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    hep.cms.text("Preliminary Simulation", ax=ax)
    if epoch is not None:
        hep.cms.lumitext(f"Epoch : {epoch}", ax=ax)

    ax.legend(*zip(*labels))
    return fig, ax

# def moveListOfTensorsToNumpy(tensors:list[torch.Tensor]) -> np.ndarray:
#     return np.concatenate([tensor.detach().cpu().numpy() for tensor in tensors])

def moveToCpu(tensor:torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu()

def convertListOfTensorsToNumpy(tensors:list[torch.Tensor]) -> np.ndarray:
    return np.concatenate([tensor.numpy() for tensor in tensors])

class SimplePlotsCallback(pl.Callback):
    def __init__(self, every_n_epochs:int=1, plotHistogramsPerEnergy=False) -> None:
        self._clearLists()
        self.every_n_epochs = every_n_epochs
        self.plotHistogramsPerEnergy = plotHistogramsPerEnergy
    
    def _shouldLog(self, current_epoch):
        # Don't log on the first epoch
        return (current_epoch - self.every_n_epochs + 1) % self.every_n_epochs == 0
    
    def _clearLists(self):
        self.tracksterEnergyEstimates = []
        self.trueBeamEnergies = []
        self.beamEnergies = []
        self.networkOutputs = []
        self.tracksterEnergies = []

    def _convertAllToNumpy(self):
        self.tracksterEnergyEstimates = convertListOfTensorsToNumpy(self.tracksterEnergyEstimates)
        self.trueBeamEnergies = convertListOfTensorsToNumpy(self.trueBeamEnergies)
        self.beamEnergies = convertListOfTensorsToNumpy(self.beamEnergies)
        self.networkOutputs = convertListOfTensorsToNumpy(self.networkOutputs)
        self.tracksterEnergies = convertListOfTensorsToNumpy(self.tracksterEnergies)

    def _plot(self, pl_module: pl.LightningModule, epochName):
        """ epochName can be Validation or Training """
        tbWriter:SummaryWriter = pl_module.logger.experiment

        tbWriter.add_figure(
            f"Scatter/{epochName}",
            scatterPredictionVsTruth(self.trueBeamEnergies, self.tracksterEnergyEstimates, 
                original_values=self.tracksterEnergies, epoch=pl_module.current_epoch),
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

        if self.plotHistogramsPerEnergy:
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
        tbWriter.add_figure(
            f"PredPerEnergy/{epochName} violin",
            doubleViolin(self.beamEnergies, self.tracksterEnergies, self.tracksterEnergyEstimates, epoch=pl_module.current_epoch, normBy="incidentBeamEnergy")[0],
            pl_module.current_epoch
        )

        fig, ax = violin(beamEnergies=self.beamEnergies, yValues=self.networkOutputs, epoch=pl_module.current_epoch)
        ax.set_ylabel("Network outputs")
        tbWriter.add_figure(f"Violin/{epochName} prediction", fig, pl_module.current_epoch)

        fig, ax = violin(beamEnergies=self.beamEnergies, yValues=pred_over_truth_array, epoch=pl_module.current_epoch)
        ax.set_ylabel("(Prediction - truth) / truth")
        tbWriter.add_figure(f"Violin/{epochName} Pred-truth / truth", fig, pl_module.current_epoch)


    def _common_batch_end(self, pl_module: pl.LightningModule, outputs:torch.Tensor, batch):
        if not self._shouldLog(pl_module.current_epoch):
            return
        self.networkOutputs.append(moveToCpu(outputs))
        self.tracksterEnergyEstimates.append(moveToCpu(pl_module.loss_params.mapNetworkOutputToEnergyEstimate(outputs, batch)))
        self.trueBeamEnergies.append(moveToCpu(batch.trueBeamEnergy))
        self.beamEnergies.append(moveToCpu(batch.beamEnergy))
        self.tracksterEnergies.append(moveToCpu(batch.tracksterEnergy))

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        if trainer.sanity_checking:  # optional skip
            return
        self._common_batch_end(pl_module, outputs["output"], batch)
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int) -> None:
        self._common_batch_end(pl_module, outputs["output"], batch)

    def _common_epoch_end(self, pl_module: pl.LightningModule, epochName:str):
        if len(self.networkOutputs) == 0:
            return
        self._convertAllToNumpy()
        self._plot(pl_module, epochName)
        self._clearLists()
    
    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # for some reason on_train_epoch_end runs after the validation epoch. So we need to deal with training metrics before the validation starts
        if trainer.sanity_checking:  # optional skip
            return
        self._common_epoch_end(pl_module, "Training")
    # def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
    #     self._common_epoch_end(pl_module, "Training")
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:  # optional skip
            return
        self._common_epoch_end(pl_module, "Validation")

    

from dataclasses import dataclass
import math
import traceback

import hist
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from torch.utils.tensorboard import SummaryWriter
import lightning.pytorch as pl
import matplotlib
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
import uncertainties

from hists.custom_hists import beamEnergiesAxis
from hists.parameters import synchrotronBeamEnergiesMap, beamEnergies
from energy_resolution.sigma_over_e import SigmaOverEComputations, fitSigmaOverE, plotSCAsEllipse, SigmaOverEPlotElement, plotSigmaOverMean, sigmaOverE_fitFunction, SigmaOverEFitError, EResolutionFitResult, plotFittedMean, plotFittedSigma

from ml.regression.drn.modules import BaseLossParameters

def makeHist():
    return hist.Hist(beamEnergiesAxis(), 
        hist.axis.Regular(bins=2000, start=0, stop=350, name="clus3D_mainTrackster_energy_pred", label="Main trackster energy, predicetd by NN (GeV)"))

@dataclass(frozen=True)
class SigmaOverEPlotElement_ML(SigmaOverEPlotElement):
    beamEnergiesInTestSet:list[int] = None
    """ List of beam energies that were placed in the test set, not used for training """


class SigmaOverEPlotter:
    def __init__(self, fit_data:str="full", overlaySigmaOverEResults:list[SigmaOverEPlotElement]=[],
            beamEnergiesInTestSet:list[int] = None,
            multiprocess_fit:str="forkserver", debug_mode=False, interactive=False) -> None:
        """ Parameters : 
         - fit_data : which dataset to use for sigma/E fitting (can be "full", or "test")
         - 
        """
        super().__init__()
        self.overlaySigmaOverEResults = overlaySigmaOverEResults
        self.fit_data = fit_data
        self.multiprocess_fit = multiprocess_fit
        self.debug_mode = debug_mode
        self.interactive = interactive
        self.beamEnergiesInTestSet = beamEnergiesInTestSet

        self.histogram_2D = makeHist()
    
    def _batchFillHistogram(self, pred_batch:torch.Tensor, data_batch:Batch, loss_params:BaseLossParameters):
        self.histogram_2D.fill(data_batch.beamEnergy.detach().cpu().numpy(), loss_params.mapNetworkOutputToEnergyEstimate(pred_batch, data_batch).detach().cpu().numpy())

    def fillHistogramFromPrediction(self, predictions:list[torch.Tensor], dataloader:DataLoader, loss_params:BaseLossParameters) -> hist.Hist:
        assert self.histogram_2D.sum() == 0, "Should start with an empty histogram"
        for pred_batch, data_batch in zip(tqdm(predictions, disable=not self.interactive, desc="Filling histogram"), dataloader, strict=True):
            self._batchFillHistogram(pred_batch, data_batch, loss_params)
        return self.histogram_2D
    
    def sigma_mu_fits(self):
        self.sigma_mu_comp = SigmaOverEComputations(recoverFromFailedFits=True)
        self.sigma_mu_results = self.sigma_mu_comp.compute(self.histogram_2D, tqdm_dict=dict(disable=True), multiprocess=self.multiprocess_fit)

    def plotSigmaOverEFits(self) -> dict[int, matplotlib.figure.Figure]:
        return {beamEnergy: self.sigma_mu_comp.plotFitResult(beamEnergy) for beamEnergy in self.sigma_mu_results.keys()}

    def fitSigmaOverE(self, useOnlyTestPoints=False) -> EResolutionFitResult:
        if useOnlyTestPoints:
            sigma_mu_results_toUse = {beamEnergy : val for beamEnergy, val in self.sigma_mu_results.items() if beamEnergy in self.beamEnergiesInTestSet}
        else:
            sigma_mu_results_toUse = self.sigma_mu_results
        self.E_res_fitResult = fitSigmaOverE(sigma_mu_results_toUse)
        self.fitUsingTestPoints = useOnlyTestPoints
        return self.E_res_fitResult
    
    @property
    def plotElt(self) -> SigmaOverEPlotElement_ML:
        try:
            fitResult = self.E_res_fitResult
            legend = "ML (fit using test points)" if self.fitUsingTestPoints else "ML"
        except AttributeError:
            fitResult = None
            legend = "ML (fit on all points)"
        return SigmaOverEPlotElement_ML(legend=legend, fitResult=fitResult, fitFunction=sigmaOverE_fitFunction, 
                dataPoints={beamEnergy : result.sigma / result.mu for beamEnergy, result in self.sigma_mu_results.items()},
                sigmaMuResults=self.sigma_mu_results,
                color="purple", beamEnergiesInTestSet=self.beamEnergiesInTestSet)


    def plotSigmaOverMean(self, **kwargs) -> matplotlib.figure.Figure:
        """ Plot sigma over E plots. All keyword arguments are passed to energy_resolution.sigma_over_e.plotSigmaOverMean """
        fig = plotSigmaOverMean(plotElements=self.overlaySigmaOverEResults + [self.plotElt], **kwargs)
        if self.plotElt.beamEnergiesInTestSet is not None:
            ax = plt.gca()
            xValues = []
            yValues = []
            for beamEnergy, yValue in self.plotElt.dataPoints.items():
                if beamEnergy in self.plotElt.beamEnergiesInTestSet:
                    xValue = synchrotronBeamEnergiesMap[beamEnergy]
                    if kwargs.get("xMode", "E") == "1/sqrt(E)":
                        xValue = 1/math.sqrt(xValue)
                    xValues.append(xValue)
                    yValues.append(yValue.nominal_value)

            ax.scatter(xValues, yValues, s=500, facecolors="none", edgecolors="red", label="Test points")
        return fig




class SigmaOverECallback(pl.Callback):
    def __init__(self, fit_data:str="full", every_n_epochs:int=5, skip_first_n_epochs:int=0, overlaySigmaOverEResults:list[SigmaOverEPlotElement]=[],
            multiprocess_fit:str="forkserver", debug_mode=False, addFitResultUsingBareData=True) -> None:
        """ Parameters : 
         - fit_data : which dataset to use for sigma/E fitting (can be "full", or "test")
         - every_n_epochs : how often to perform the fits
         - skip_first_n_epochs : don't fit for the given number of epochs
        """
        super().__init__()
        
        self.every_n_epochs = every_n_epochs
        self.skip_first_n_epochs = skip_first_n_epochs
        self.overlaySigmaOverEResults = overlaySigmaOverEResults
        self.fit_data = fit_data
        self.multiprocess_fit = multiprocess_fit
        self.debug_mode = debug_mode
        self.addFitResultUsingBareData = addFitResultUsingBareData
    
    def _fillHistogram(self, sigma_over_e_dataloader:DataLoader, pl_module: pl.LightningModule, plotter:SigmaOverEPlotter):
        for i, batch in enumerate(sigma_over_e_dataloader):
            batch_device = pl_module.transfer_batch_to_device(batch, pl_module.device, dataloader_idx=0)
            pred = pl_module.predict_step(batch_device, i)

            plotter._batchFillHistogram(pred, batch_device, loss_params=pl_module.loss_params)
        
        return plotter.histogram_2D
    
    def _getDataloader(self, trainer:pl.Trainer) -> DataLoader:
        if self.fit_data == "test":
            return trainer.datamodule.test_dataloader()
        elif self.fit_data == "full":
            return trainer.datamodule.full_dataloader()
        else:
            raise ValueError("Wrong fit_data")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch < self.skip_first_n_epochs or (trainer.current_epoch - self.skip_first_n_epochs) % self.every_n_epochs != 0:
            return
        if trainer.sanity_checking:  # optional skip
            return
        if not self.debug_mode and self.every_n_epochs > 1 and trainer.current_epoch == 0:
            return # don't fit on first epoch
        print("Start predicting!")
        try:
            dataloader = self._getDataloader(trainer)
            
            self.sigmaOverEPlotter = SigmaOverEPlotter(fit_data=self.fit_data, overlaySigmaOverEResults=self.overlaySigmaOverEResults,
                multiprocess_fit=self.multiprocess_fit, beamEnergiesInTestSet=trainer.datamodule.dataset_test.beamEnergiesToSelect)
            h = self._fillHistogram(dataloader, pl_module, self.sigmaOverEPlotter)

            print("Start fitting!")
            tbWriter:SummaryWriter = pl_module.logger.experiment
            
            # use forkserver multiprocessing start mode to avoid fork issues with PyTorch/Tensorflow/CUDA
            self.sigmaOverEPlotter.sigma_mu_fits()
            if self.debug_mode:
                for beamEnergy, fig in self.sigmaOverEPlotter.plotSigmaOverEFits().items():
                    tbWriter.add_figure(
                        f"Debug/GaussianFit-{beamEnergy}",
                        fig,
                        trainer.current_epoch
                    )
            
            tbWriter.add_figure(
                "SigmaOverE/Full simulation (fct of E)",
                plotSigmaOverMean(self.overlaySigmaOverEResults + [self.sigmaOverEPlotter.plotElt], xMode="E"),
                trainer.current_epoch)
            
            tbWriter.add_figure(
                "FittedMeanSigma/Mean - Full simulation",
                plotFittedMean(self.overlaySigmaOverEResults + [self.sigmaOverEPlotter.plotElt], errors=True, beamEnergiesToCircle=True),
                trainer.current_epoch
            )
            tbWriter.add_figure(
                "FittedMeanSigma/Sigma - Full simulation",
                plotFittedSigma(self.overlaySigmaOverEResults + [self.sigmaOverEPlotter.plotElt], normBy="sqrt(E)", errors=True, beamEnergiesToCircle=True),
                trainer.current_epoch
            )
            
            plotFit = True
            try:
                fitRes = self.sigmaOverEPlotter.fitSigmaOverE()
                straightLineFit_exception = None

                pl_module.log_dict({
                    "EnergyResolution/C (full simulation)":fitRes.C.nominal_value,
                    "EnergyResolution/S (full simulation)":fitRes.S.nominal_value,
                    "EnergyResolution/S*C (full simulation)":fitRes.S.nominal_value*fitRes.C.nominal_value})
                
                tbWriter.add_figure("SigmaOverE/Full simulation (ellipse)",
                    plotSCAsEllipse(self.overlaySigmaOverEResults + [self.sigmaOverEPlotter.plotElt]),
                    trainer.current_epoch)
                
            except Exception as e:
                straightLineFit_exception = e
                plotFit = False
                pass # in case the straight line fit fails : still plot without the fit
            
            tbWriter.add_figure(
                "SigmaOverE/Full simulation (fct of 1/sqrt(E))",
                self.sigmaOverEPlotter.plotSigmaOverMean(xMode="1/sqrt(E)", plotFit=plotFit),
                trainer.current_epoch)
            
            ## Redo the fit using only test points
            try:
                fitRes = self.sigmaOverEPlotter.fitSigmaOverE(useOnlyTestPoints=True)

                pl_module.log_dict({
                    "EnergyResolution/C (test set)":fitRes.C.nominal_value,
                    "EnergyResolution/S (test set)":fitRes.S.nominal_value,
                    "EnergyResolution/S*C (test set)":fitRes.S.nominal_value*fitRes.C.nominal_value})
                
                tbWriter.add_figure(
                    "SigmaOverE/Fit on test set only (fct of 1/sqrt(E))",
                    self.sigmaOverEPlotter.plotSigmaOverMean(xMode="1/sqrt(E)", plotFit=True),
                    trainer.current_epoch)

                tbWriter.add_figure("SigmaOverE/Fit on test set only (ellipse)",
                    plotSCAsEllipse(self.overlaySigmaOverEResults + [self.sigmaOverEPlotter.plotElt]),
                    trainer.current_epoch)

            except Exception as e:
                straightLineFit_exception = e
                pass
            
            if straightLineFit_exception is not None:
                raise straightLineFit_exception

        except (RuntimeError, SigmaOverEFitError) as e:
            print("SigmaOverE fit failed due to : " + str(e))
            if self.debug_mode:
                raise e
            else:
                print(e)
                print(traceback.format_exc())
        except Exception as e:
            print("SigmaOverE fit failed due to unknown exception : ")
            if self.debug_mode:
                raise e
            else:
                print(e)
                print(traceback.format_exc())


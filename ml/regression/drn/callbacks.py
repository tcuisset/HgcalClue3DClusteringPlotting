import hist
import torch
from torch_geometric.loader import DataLoader
import lightning.pytorch as pl
import matplotlib
from tqdm.auto import tqdm

from hists.custom_hists import beamEnergiesAxis
from energy_resolution.sigma_over_e import SigmaOverEComputations, fitSigmaOverE, plotSCAsEllipse, SigmaOverEPlotElement, plotSigmaOverMean, sigmaOverE_fitFunction, SigmaOverEFitError, EResolutionFitResult


def makeHist():
    return hist.Hist(beamEnergiesAxis(), 
        hist.axis.Regular(bins=2000, start=0, stop=350, name="clus3D_mainTrackster_energy_pred", label="Main trackster energy, predicetd by NN (GeV)"))


class SigmaOverEPlotter:
    def __init__(self, prediction_type:str, fit_data:str="full", overlaySigmaOverEResults:list[SigmaOverEPlotElement]=[],
            multiprocess_fit:str="forkserver", debug_mode=False, interactive=False) -> None:
        """ Parameters : 
         - prediction_type : can be "absolute" (prediction is estimate of energy in GeV) or "ratio" (pred is ratio estimate energy/gun energy)
         - fit_data : which dataset to use for sigma/E fitting (can be "full", or "test")
         - 
        """
        super().__init__()
        self.prediction_type = prediction_type
        self.overlaySigmaOverEResults = overlaySigmaOverEResults
        self.fit_data = fit_data
        self.multiprocess_fit = multiprocess_fit
        self.debug_mode = debug_mode
        self.interactive = interactive

        self.histogram_2D = makeHist()
    
    def _batchFillHistogram(self, pred_batch:torch.Tensor, data_batch:torch.Tensor):
        if self.prediction_type == "absolute":
            self.histogram_2D.fill(data_batch.beamEnergy.detach().cpu().numpy(), pred_batch.detach().cpu().numpy())
        elif self.prediction_type == "ratio":
            self.histogram_2D.fill(data_batch.beamEnergy.detach().cpu().numpy(), (pred_batch*data_batch.trueBeamEnergy).detach().cpu().numpy())
        else:
            raise ValueError(f"SigmaOverECallback : prediction_type not supported : {self.prediction_type}")

    def fillHistogramFromPrediction(self, predictions:list[torch.Tensor], dataloader:DataLoader) -> hist.Hist:
        assert self.histogram_2D.sum() == 0, "Should start with an empty histogram"
        for pred_batch, data_batch in zip(tqdm(predictions, disable=not self.interactive), dataloader, strict=True):
            self._batchFillHistogram(pred_batch, data_batch)
        return self.histogram_2D
    
    def sigma_mu_fits(self):
        self.sigma_mu_comp = SigmaOverEComputations(recoverFromFailedFits=True)
        self.sigma_mu_results = self.sigma_mu_comp.compute(self.histogram_2D, tqdm_dict=dict(disable=True), multiprocess=self.multiprocess_fit)

    def plotSigmaOverEFits(self) -> dict[int, matplotlib.figure.Figure]:
        return {beamEnergy: self.sigma_mu_comp.plotFitResult(beamEnergy) for beamEnergy in self.sigma_mu_results.keys()}

    def plotSigmaOverE_debug(self) -> matplotlib.figure.Figure:
        return plotSigmaOverMean(self.overlaySigmaOverEResults + [self.plotElt], plotFit=False)

    def fitSigmaOverE(self) -> EResolutionFitResult:
        self.E_res_fitResult = fitSigmaOverE(self.sigma_mu_results)
        return self.E_res_fitResult
    
    @property
    def plotElt(self) -> SigmaOverEPlotElement:
        try:
            fitResult = self.E_res_fitResult
        except AttributeError:
            fitResult = None
        return SigmaOverEPlotElement(legend="ML", fitResult=fitResult, fitFunction=sigmaOverE_fitFunction, 
                dataPoints={beamEnergy : result.sigma / result.mu for beamEnergy, result in self.sigma_mu_results.items()}, color="green")
    
class SigmaOverECallback(pl.Callback):
    def __init__(self, prediction_type:str, fit_data:str="full", every_n_epochs:int=100, overlaySigmaOverEResults:list[SigmaOverEPlotElement]=[],
            multiprocess_fit:str="forkserver", debug_mode=False) -> None:
        """ Parameters : 
         - prediction_type : can be "absolute" (prediction is estimate of energy in GeV) or "ratio" (pred is ratio estimate energy/gun energy)
         - fit_data : which dataset to use for sigma/E fitting (can be "full", or "test")
         - 
        """
        super().__init__()
        
        self.prediction_type = prediction_type
        self.every_n_epochs = every_n_epochs
        self.overlaySigmaOverEResults = overlaySigmaOverEResults
        self.fit_data = fit_data
        self.multiprocess_fit = multiprocess_fit
        self.debug_mode = debug_mode
    
    def _fillHistogram(self, sigma_over_e_dataloader:DataLoader, pl_module: pl.LightningModule, plotter:SigmaOverEPlotter):
        for i, batch in enumerate(sigma_over_e_dataloader):
            batch_device = pl_module.transfer_batch_to_device(batch, pl_module.device, dataloader_idx=0)
            pred = pl_module.predict_step(batch_device, i)

            plotter._batchFillHistogram(pred, batch_device)
        
        return plotter.histogram_2D
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        if trainer.sanity_checking:  # optional skip
            return
        if not self.debug_mode and self.every_n_epochs > 1 and trainer.current_epoch == 0:
            return # don't fit on first epoch
        print("Start predicting!")
        try:
            if self.fit_data == "test":
                dataloader = trainer.datamodule.test_dataloader()
            elif self.fit_data == "full":
                dataloader = trainer.datamodule.full_dataloader()
            else:
                raise ValueError("Wrong fit_data")
            
            self.sigmaOverEPlotter = SigmaOverEPlotter(prediction_type=self.prediction_type, fit_data=self.fit_data, overlaySigmaOverEResults=self.overlaySigmaOverEResults,
                multiprocess_fit=self.multiprocess_fit)
            h = self._fillHistogram(dataloader, pl_module, self.sigmaOverEPlotter)

            print("Start fitting!")
            
            # use forkserver multiprocessing start mode to avoid fork issues with PyTorch/Tensorflow/CUDA
            self.sigmaOverEPlotter.sigma_mu_fits()
            if self.debug_mode:
                for beamEnergy, fig in self.sigmaOverEPlotter.plotSigmaOverEFits():
                    pl_module.logger.experiment.add_figure(
                        f"Debug/GaussianFit-{beamEnergy}",
                        fig,
                        trainer.current_epoch
                    )
            
            pl_module.logger.experiment.add_figure(
                "Validation/SigmaOverE (fct of E)",
                plotSigmaOverMean(self.overlaySigmaOverEResults + [self.sigmaOverEPlotter.plotElt], xMode="E"),
                trainer.current_epoch)
            
            try:
                self.sigmaOverEPlotter.fitSigmaOverE()
                straightLineFit_exception = None
            except Exception as e:
                straightLineFit_exception = e
                pass # in case the straight line fit fails : still plot without the fit
            
            pl_module.logger.experiment.add_figure(
                "Validation/SigmaOverE (fct of 1/sqrt(E))",
                plotSigmaOverMean(self.overlaySigmaOverEResults + [self.sigmaOverEPlotter.plotElt], xMode="1/sqrt(E)", plotFit=True),
                trainer.current_epoch)

            pl_module.logger.experiment.add_figure("Validation/SigmaOverE (ellipse)",
                plotSCAsEllipse(self.overlaySigmaOverEResults + [self.sigmaOverEPlotter.plotElt]),
                trainer.current_epoch)
            
            if straightLineFit_exception is not None:
                raise straightLineFit_exception

        except (RuntimeError, SigmaOverEFitError) as e:
            print("SigmaOverE fit failed due to : " + str(e))
        except Exception as e:
            print("SigmaOverE fit failed due to unknown exception : ")
            if self.debug_mode:
                raise e
            else:
                print(e)


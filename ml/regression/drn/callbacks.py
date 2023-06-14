import hist
from torch_geometric.loader import DataLoader
import lightning.pytorch as pl

from hists.custom_hists import beamEnergiesAxis
from energy_resolution.sigma_over_e import SigmaOverEComputations, fitSigmaOverE, plotSCAsEllipse, SigmaOverEPlotElement, plotSigmaOverMean, sigmaOverE_fitFunction, SigmaOverEFitError


def makeHist():
    return hist.Hist(beamEnergiesAxis(), 
        hist.axis.Regular(bins=2000, start=0, stop=350, name="clus3D_mainTrackster_energy_pred", label="Main trackster energy, predicetd by NN (GeV)"))


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
    
    def _fillHistogram(self, sigma_over_e_dataloader:DataLoader, pl_module: pl.LightningModule):
        h = makeHist()
        for i, batch in enumerate(sigma_over_e_dataloader):
            batch_device = pl_module.transfer_batch_to_device(batch, pl_module.device, dataloader_idx=0)
            pred = pl_module.predict_step(batch_device, i)
            if self.prediction_type == "absolute":
                h.fill(batch.beamEnergy.detach().cpu().numpy(), pred.detach().cpu().numpy())
            elif self.prediction_type == "ratio":
                h.fill(batch.beamEnergy.detach().cpu().numpy(), (pred*batch_device.trueBeamEnergy).detach().cpu().numpy())
            else:
                raise ValueError(f"SigmaOverECallback : prediction_type not supported : {self.prediction_type}")
        
        return h
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        if trainer.sanity_checking:  # optional skip
            return
        
        print("Start predicting!")
        try:
            if self.fit_data == "test":
                dataloader = trainer.datamodule.test_dataloader()
            elif self.fit_data == "full":
                dataloader = trainer.datamodule.full_dataloader()
            else:
                raise ValueError("Wrong fit_data")
            h = self._fillHistogram(dataloader, pl_module)

            print("Start fitting!")
            
            # use forkserver multiprocessing start mode to avoid fork issues with PyTorch/Tensorflow/CUDA
            sigma_mu_comp = SigmaOverEComputations(recoverFromFailedFits=True)
            sigma_mu_results = sigma_mu_comp.compute(h, tqdm_dict=dict(disable=True), multiprocess=self.multiprocess_fit)
            if self.debug_mode:
                for beamEnergy in sigma_mu_results.keys():
                    pl_module.logger.experiment.add_figure(
                        f"Debug/GaussianFit-{beamEnergy}",
                        sigma_mu_comp.plotFitResult(beamEnergy),
                        trainer.current_epoch
                    )
                pl_module.logger.experiment.add_figure(
                    f"Debug/SigmeOverMu-nofit",
                    plotSigmaOverMean([SigmaOverEPlotElement(legend="ML", dataPoints={beamEnergy : result.sigma / result.mu for beamEnergy, result in sigma_mu_results.items()})],
                        plotFit=False, sim=True),
                    trainer.current_epoch
                )
                

            E_res_fitResult = fitSigmaOverE(sigma_mu_results)
            plotElt = SigmaOverEPlotElement(legend="ML", fitResult=E_res_fitResult, fitFunction=sigmaOverE_fitFunction, 
                dataPoints={beamEnergy : result.sigma / result.mu for beamEnergy, result in sigma_mu_results.items()}, color="green")
            
            pl_module.logger.experiment.add_figure(
                "Validation/SigmaOverE (fct of E)",
                plotSigmaOverMean(self.overlaySigmaOverEResults + [plotElt], xMode="E"),
                trainer.current_epoch)
            pl_module.logger.experiment.add_figure(
                "Validation/SigmaOverE (fct of 1/sqrt(E))",
                plotSigmaOverMean(self.overlaySigmaOverEResults + [plotElt], xMode="1/sqrt(E)"),
                trainer.current_epoch)

            pl_module.logger.experiment.add_figure("Validation/SigmaOverE (ellipse)",
                plotSCAsEllipse(self.overlaySigmaOverEResults + [plotElt]),
                trainer.current_epoch)

        except (RuntimeError, SigmaOverEFitError) as e:
            print("SigmaOverE fit failed due to : " + str(e))
        except Exception as e:
            print("SigmaOverE fit failed due to unknown exception : ")
            if self.debug_mode:
                raise e
            else:
                print(e)


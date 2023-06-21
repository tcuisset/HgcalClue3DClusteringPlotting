from functools import partial
import traceback
import sys
sys.path.append("../../..")
sys.path.append("/grid_mnt/vol_home/llr/cms/cuisset/hgcal/testbeam18/clue3d-dev/src/Plotting")

import matplotlib.pyplot as plt
import mplhep as hep

import lightning.pytorch as pl
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI, SaveConfigCallback
from torch.utils.tensorboard import SummaryWriter
from jsonargparse import Namespace

from energy_resolution.sigma_over_e import plotSCAsEllipse, plotSigmaOverMean, SigmaOverEFitError
from ml.regression.drn.modules import *
from ml.regression.drn.callbacks.sigma_over_e import SigmaOverECallback, SigmaOverEPlotter
from ml.regression.drn.predict.predict import Predictor


##### Backport
# Until https://github.com/Lightning-AI/lightning/pull/17475 is released, this is a temporary workaround based on this PR's code
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.fabric.utilities.cloud_io import get_filesystem
class SaveConfigCallback_custom(pl.Callback):
    """Saves a LightningCLI config to the log_dir when training starts.

    Args:
        parser: The parser object used to parse the configuration.
        config: The parsed configuration that will be saved.
        config_filename: Filename for the config file.
        overwrite: Whether to overwrite an existing config file.
        multifile: When input is multiple config files, saved config preserves this structure.
        save_to_log_dir: Whether to save the config to the log_dir.

    Raises:
        RuntimeError: If the config file already exists in the directory to avoid overwriting a previous run
    """

    def __init__(
        self,
        parser: LightningArgumentParser,
        config: Namespace,
        config_filename: str = "config.yaml",
        overwrite: bool = False,
        multifile: bool = False,
        save_to_log_dir: bool = True,
    ) -> None:
        self.parser = parser
        self.config = config
        self.config_filename = config_filename
        self.overwrite = overwrite
        self.multifile = multifile
        self.save_to_log_dir = save_to_log_dir
        self.already_saved = False

        if not save_to_log_dir and not is_overridden("save_config", self, SaveConfigCallback):
            raise ValueError(
                "`save_to_log_dir=False` only makes sense when subclassing SaveConfigCallback to implement "
                "`save_config` and it is desired to disable the standard behavior of saving to log_dir."
            )

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        if self.already_saved:
            return

        if self.save_to_log_dir:
            log_dir = trainer.log_dir  # this broadcasts the directory
            assert log_dir is not None
            config_path = os.path.join(log_dir, self.config_filename)
            fs = get_filesystem(log_dir)

            if not self.overwrite:
                # check if the file exists on rank 0
                file_exists = fs.isfile(config_path) if trainer.is_global_zero else False
                # broadcast whether to fail to all ranks
                file_exists = trainer.strategy.broadcast(file_exists)
                if file_exists:
                    raise RuntimeError(
                        f"{self.__class__.__name__} expected {config_path} to NOT exist. Aborting to avoid overwriting"
                        " results of a previous run. You can delete the previous config file,"
                        " set `LightningCLI(save_config_callback=None)` to disable config saving,"
                        ' or set `LightningCLI(save_config_kwargs={"overwrite": True})` to overwrite the config file.'
                    )

            if trainer.is_global_zero:
                # save only on rank zero to avoid race conditions.
                # the `log_dir` needs to be created as we rely on the logger to do it usually
                # but it hasn't logged anything at this point
                fs.makedirs(log_dir, exist_ok=True)
                self.parser.save(
                    self.config, config_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
                )

        if trainer.is_global_zero:
            self.save_config(trainer, pl_module, stage)
            self.already_saved = True

        # broadcast so that all ranks are in sync on future calls to .setup()
        self.already_saved = trainer.strategy.broadcast(self.already_saved)

    def save_config(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Implement to save the config in some other place additional to the standard log_dir.

        Example:
            def save_config(self, trainer, pl_module, stage):
                if isinstance(trainer.logger, Logger):
                    config = self.parser.dump(self.config, skip_none=False)  # Required for proper reproducibility
                    trainer.logger.log_hyperparams({"config": config})

        Note:
            This method is only called on rank zero. This allows to implement a custom save config without having to
            worry about ranks or race conditions. Since it only runs on rank zero, any collective call will make the
            process hang waiting for a broadcast. If you need to make collective calls, implement the setup method
            instead.
        """

class SaveHyperparametersConfigCallback(SaveConfigCallback_custom):
    """ Save hyperparameters to logger
    """
    def __init__(self, parser: LightningArgumentParser, config: Namespace, config_filename: str = "config.yaml", overwrite: bool = False, multifile: bool = False, save_to_log_dir: bool = True,
                 metrics_keys:list=[]) -> None:
        """ metrics_names : list of metrics to log into hyperparamaters """
        super().__init__(parser, config, config_filename, overwrite, multifile, save_to_log_dir)
        self.metrics_dict = {name : 0 for name in metrics_keys}

    def _addKeys(self, input_namespace:Namespace, keys:list[str], prefix:str=""):
        out_dict = dict()
        for key in keys:
            try:
                out_dict[prefix + key] = input_namespace[key]
            except KeyError:
                pass
        return out_dict

    def save_config(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        hp = dict()
        try:
            hp.update(pl_module.hyperparameters)
        except AttributeError:
            pass

        hp["optimizer"] = self.config.model.optimizer.class_path
        hp.update(self._addKeys(self.config.model.optimizer.init_args, ["lr", "weight_decay"], "optimizer."))

        hp["lr_scheduler"] = self.config.model.lr_scheduler.class_path
        hp.update(self._addKeys(self.config.model.lr_scheduler.init_args, 
            ["restart_period", "t_mult", "policy", "min_lr", "gamma", "triangular_step"], "lr_scheduler."))

        hp["loss"] = self.config.model.loss.class_path
        pl_module.hyperparameters = hp
        #trainer.logger.log_hyperparams(hp, self.metrics_dict)
        #trainer.logger.hparams.update(hp)

class DRNCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_lightning_class_args(SigmaOverECallback, "sigma_over_e_callback")
        parser.link_arguments("data.reader", "sigma_over_e_callback.overlaySigmaOverEResults",
            lambda reader : [reader.loadSigmaOverEResults("rechits"), reader.loadSigmaOverEResults("clue3d")], apply_on="instantiate")

def cli_main(args = None, metrics_keys:list=[], **cli_kwargs):
    cli = DRNCLI(DRNModule, DRNDataModule, args=args, auto_configure_optimizers=False,
         save_config_callback=partial(SaveHyperparametersConfigCallback, metrics_keys=metrics_keys), **cli_kwargs)
    return cli



class DataMetricsLogger:
    def __init__(self, cli:LightningCLI, ckpt_path:str, data_trainer_kwargs:dict=dict()) -> None:
        self.cli = cli
        self.ckpt_path = ckpt_path
        self.data_trainer_kwargs = data_trainer_kwargs
        self.epochPicked = -1 # TODO
        self.scalarMetrics = dict()

    def predict(self):
        self.data_predictor = Predictor(self.cli.model, datasetComputationClass=self.cli.datamodule.datasetComputationClass)
        self.data_predictor.loadDataDataset()
        trainer_kwargs = dict(logger=False, enable_checkpointing=False)
        trainer_kwargs.update(self.data_trainer_kwargs)

        trainer_kwargs = {**self.cli._get(self.cli.config_init, "trainer", default={}), **trainer_kwargs}

        # Remove all callbacks except our predict callback (otherwise there are SaveConfigCallbacks that run, etc)
        trainer_kwargs["callbacks"] = [self.data_predictor.getPredictCallback()]
        trainer = self.cli.trainer_class(**trainer_kwargs)
        self.data_predictor.runPredictionOnTrainer(trainer, ckpt_path=self.ckpt_path)
    
    def makePlotter(self):
        self.overlaySigmaOverEResults = [self.data_predictor.reader.loadSigmaOverEResults("rechits"), self.data_predictor.reader.loadSigmaOverEResults("clue3d")]
        self.plotter = SigmaOverEPlotter(overlaySigmaOverEResults=self.overlaySigmaOverEResults, 
            interactive=False, beamEnergiesInTestSet=self.cli.datamodule.dataset_test.beamEnergiesToSelect)
        self.plotter.histogram_2D = self.data_predictor.h_2D

    def sigmaMuFits(self):
        #self._logScalarMetric({"test_metric" : 5.})
        self.plotter.sigma_mu_fits()

    
    def _logScalarMetric(self, d:dict):
        logger = self.cli.trainer.logger
        logger.log_metrics(d, self.epochPicked)
        self.scalarMetrics.update(d)

    def logMetrics(self):
        logger = self.cli.trainer.logger
        tbWriter:SummaryWriter = logger.experiment

        plt.style.use(hep.style.CMS)

        tbWriter.add_figure(
            "SigmaOverE/Full simulation (fct of E)",
            plotSigmaOverMean(self.overlaySigmaOverEResults + [self.plotter.plotElt], xMode="E"),
            self.epochPicked)
        
        exceptions:list[tuple[str, Exception]] = []
        try:
            fitRes = self.plotter.fitSigmaOverE()

            self._logScalarMetric({
                "EnergyResolution/C (full data)":fitRes.C.nominal_value,
                "EnergyResolution/S (full data)":fitRes.S.nominal_value,
                "EnergyResolution/S*C (full data)":fitRes.S.nominal_value*fitRes.C.nominal_value})
            
            tbWriter.add_figure("SigmaOverE/Full data (ellipse)",
                plotSCAsEllipse(self.overlaySigmaOverEResults + [self.plotter.plotElt]),
                self.epochPicked)
            
        except Exception as e:
            exceptions.append(("Energy resolution straight line fit error (using full data)", e))
            pass # in case the straight line fit fails : still plot without the fit
        
        tbWriter.add_figure(
            "SigmaOverE/Full data (fct of 1/sqrt(E))",
            self.plotter.plotSigmaOverMean(xMode="1/sqrt(E)", plotFit=True),
            self.epochPicked)

        
        ## Redo the fit using only test points
        try:
            fitRes = self.plotter.fitSigmaOverE(useOnlyTestPoints=True)

            self._logScalarMetric({
                "EnergyResolution/C (test set)":fitRes.C.nominal_value,
                "EnergyResolution/S (test set)":fitRes.S.nominal_value,
                "EnergyResolution/S*C (test set)":fitRes.S.nominal_value*fitRes.C.nominal_value})
            
            tbWriter.add_figure(
                "SigmaOverE/Fit on test set only (fct of 1/sqrt(E))",
                self.plotter.plotSigmaOverMean(xMode="1/sqrt(E)", plotFit=True),
                self.epochPicked)

            tbWriter.add_figure("SigmaOverE/Fit on test set only (ellipse)",
                plotSCAsEllipse(self.overlaySigmaOverEResults + [self.plotter.plotElt]),
                self.epochPicked)

        except Exception as e:
            exceptions.append(("Energy resolution straight line fit error (using test data only)", e))
            pass
        
        if len(exceptions) > 0:
            for message, e in exceptions:
                print(message)
                print(e)
                print("")

    def makeAll(self):
        self.predict()
        self.makePlotter()
        try:
            self.sigmaMuFits()
            self.logMetrics()
        except SigmaOverEFitError as e:
            print(traceback.format_exc())
     


if __name__ == "__main__":
    cli_main()
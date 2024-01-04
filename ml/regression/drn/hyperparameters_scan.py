""" Script for scanning for hyperparameters. 
All command line arguments are forwarded to LightningCLI (you should set common config)
"""
from collections import namedtuple
from dataclasses import dataclass
import itertools
import typing
import random
import sys
import traceback
sys.path.append("/grid_mnt/vol_home/llr/cms/cuisset/hgcal/testbeam18/clue3d-dev/src/Plotting")

import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI

from ml.regression.drn.cli import cli_main, DataMetricsLogger

@dataclass
class SingleParameterConfig:
    label:str|None
    """ A label for this configuration (facultative), used for experiment naming """
    config:dict[str, typing.Any]
    """ The actual configuration to try, as a dict """
    parentName:str|None = None
    """ The name of the parent scan, ie the ParameterScan holding this SingleParameterConfig """

@dataclass
class ParameterScan:
    name:str|None
    """ Name of the scan, for information only """
    configs:list[SingleParameterConfig]
    """ List of all different configurations to try for this parameter scan """


def makeSimpleScan(key:str, values:list) -> list[dict[str, typing.Any]]:
    return [SingleParameterConfig(label=str(value), config={key:value}) for value in values]

parametersForScan_LC = [
    ParameterScan(name="dropout", configs=makeSimpleScan("model.drn.init_args.dropout", [0.05, 0.1]#[0.1, 0.2, 0.3]
                                                         )),
    # ParameterScan(name="hidden_dim", configs=makeSimpleScan("model.drn.init_args.hidden_dim",  [40, 60, 80, 100]#[15, 20, 25]
    #                                                         )),
    # ParameterScan(name="k", configs=makeSimpleScan("model.drn.init_args.k",  [5, 10, 20, 30]#[5, 10, 20]
    #                                                )),
    # ParameterScan(name="lr_scheduler", configs=makeSimpleScan("model.lr_scheduler.restart_period", [50]#[20, 80, 130]
    #                                                           )),
    # ParameterScan(name="loss", configs=[
    #     SingleParameterConfig("RatioCorrectedLoss", 
    #         {"model.loss": "ml.regression.drn.modules.RatioCorrectedLoss",
    #         "model.loss.coefs" : [-0.2597882 , -0.24326517,  1.01537901]}),
    #     # SingleParameterConfig("RatioRelativeMSE",
    #     #     {"model.loss": "ml.regression.drn.modules.RatioRelativeMSE"}),
    #     # SingleParameterConfig("RatioRelativeExpLoss",
    #     # {"model.loss": "ml.regression.drn.modules.RatioRelativeExpLoss"}),
    # ]),
    # ParameterScan(name="drn_norm", configs=[
    #     #SingleParameterConfig("all-ones", {"model.drn.norm" : [1., 1., 1., 1.]}),
    #     #SingleParameterConfig("v1", {"model.drn.norm" : [1./(2*6.8), 1./(2*6.8), 1./28, 1.]}),
    #     #SingleParameterConfig("E10", {"model.drn.norm" : [1./(2*6.8), 1./(2*6.8), 1./28, 10.]}),
    #     SingleParameterConfig("E50", {"model.drn.norm" : [1./(2*6.8), 1./(2*6.8), 1./28, 50.]}),
    #     SingleParameterConfig("E100", {"model.drn.norm" : [1./(2*6.8), 1./(2*6.8), 1./28, 100.]}),
    # ])

]

parametersForScan_rechits = [
    ParameterScan(name="dropout", configs=makeSimpleScan("model.drn.init_args.dropout", [0.1]
                                                         )),
    ParameterScan(name="hidden_dim", configs=makeSimpleScan("model.drn.init_args.hidden_dim",  [50]#[15, 20, 25]
                                                            )),
    ParameterScan(name="k", configs=makeSimpleScan("model.drn.init_args.k",  [5]#[5, 10, 20]
                                                   )),
    ParameterScan(name="lr_scheduler", configs=makeSimpleScan("model.lr_scheduler.restart_period", [30]#[20, 80, 130]
                                                              )),
    # ParameterScan(name="loss", configs=[
    #     SingleParameterConfig("RatioCorrectedLoss", 
    #         {"model.loss": "ml.regression.drn.modules.RatioCorrectedLoss",
    #         "model.loss.coefs" : [-0.2597882 , -0.24326517,  1.01537901]}),
    #     # SingleParameterConfig("RatioRelativeMSE",
    #     #     {"model.loss": "ml.regression.drn.modules.RatioRelativeMSE"}),
    #     # SingleParameterConfig("RatioRelativeExpLoss",
    #     # {"model.loss": "ml.regression.drn.modules.RatioRelativeExpLoss"}),
    # ])
]


parametersForScan = parametersForScan_LC
hyperparameters_metric_keys = ["EnergyResolution/S*C (test set)", "EnergyResolution/S (test set)", "EnergyResolution/C (test set)",
    "EnergyResolution/S*C (full data)", "EnergyResolution/S (full data)", "EnergyResolution/C (full data)", # for training on data
    "EnergyResolution-data/C (full data)", "EnergyResolution-data/S (full data)", "EnergyResolution-data/S*C (full data)"]
""" List of metrics keys to save in hyperparameters view """
   

def runForSingleParameterSet(args, data_trainer_kwargs:dict=dict(), hyperparameters_metric_keys=hyperparameters_metric_keys):
    cli = cli_main(args, run=False)
    modelCheckpoint = cli.trainer.checkpoint_callback
    if not cli.datamodule.reader.isSimulation:
        cli.trainer.logger.log_hyperparams(cli.model.hyperparameters, {key : -1 for key in hyperparameters_metric_keys})
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)
    if cli.datamodule.reader.isSimulation:
        try:
            metricsLogger = DataMetricsLogger(cli, ckpt_path=modelCheckpoint.best_model_path, data_trainer_kwargs=data_trainer_kwargs)
            metricsLogger.makeAll()
        except:
            print(traceback.format_exc())

        # Saving hyperparameters
        try:
            metrics_to_save = {key:value for key, value in metricsLogger.scalarMetrics.items() if key in hyperparameters_metric_keys}
        except:
            metrics_to_save = {}
    
        cli.trainer.logger.log_hyperparams(cli.model.hyperparameters, metrics_to_save)
        cli.trainer.logger.save()
    return cli

def scan(parameters:list[ParameterScan], default_args=None):
    if default_args is None:
        default_args = sys.argv[1:]
        sys.argv = sys.argv[:1] # dont pass argv to LightningCLI

    list_of_lists = []
    for scan in parameters:
        inner_list = []
        for singleParamConfig in scan.configs:
            singleParamConfig.parentName = scan.name
            inner_list.append(singleParamConfig)
        list_of_lists.append(inner_list)

    # make a list of parameters (outer list : param name) of individual config
    #list_of_lists = [[SingleParameterConfig(parameterScan.name, config) for config in parameterScan.configs] for parameterScan in parameters]
    cartesian = list(itertools.product(*list_of_lists)) # explode it into cartesian product
    random.shuffle(cartesian)
    for parameterConfigs in cartesian:
        # parameterConfigs is a list of SingleParameterConfig

        args = []
        experimentVersionParts = []
        #args = [f"--{key}={value}" for singleParamConfig in parameterConfigs for key, value in singleParamConfig.config.items()]
        #print(default_args+args)

        for singleParamConfig in parameterConfigs:
            args += [f"--{key}={value}" for key, value in singleParamConfig.config.items()]
            if singleParamConfig.parentName is not None and singleParamConfig.label is not None:
                experimentVersionParts.append(f"{singleParamConfig.parentName}={singleParamConfig.label}")

        if len(experimentVersionParts) > 0:
            experimentVersion = "-".join(experimentVersionParts)
            #args.append(f"--trainer.logger=TensorBoardLogger")
            args.append(f"--trainer.logger.version={experimentVersion}") 

        try:
            oom_error = False
            try:
                runForSingleParameterSet(args=default_args+args)
            except RuntimeError as e:
                print("RuntimeError occurred, will try again with reduced batch_size")
                print(e)
                print(traceback.format_exc())

                oom_error = True
                # see https://pytorch.org/docs/stable/notes/faq.html#my-out-of-memory-exception-handler-can-t-allocate-memory
                # don't rerun the training while the exception is in scope
            if oom_error:
                runForSingleParameterSet(args=default_args+args+["--data.batch_size=512", f"--trainer.logger.version={experimentVersion}-OOMRestart"])
        except Exception as e:
            print(traceback.format_exc())

        

if __name__ == "__main__":
    scan(parameters=parametersForScan)
import hist
import lightning.pytorch as pl
from torch_geometric.loader import DataLoader

from ntupleReaders.clue_ntuple_reader import ClueNtupleReader
from energy_resolution.sigma_over_e import plotSigmaOverMean

from ml.regression.drn.modules import DRNModule, DRNDataset
from ml.regression.drn.dataset_making import LayerClustersTensorMaker
from ml.regression.drn.callbacks.sigma_over_e import SigmaOverEPlotter, makeHist

class Predictor:
    def __init__(self, model:DRNModule, datasetComputationClass, batch_size=1024) -> None:
        self.model = model
        self.datasetComputationClass = datasetComputationClass
        self.batch_size = batch_size
        self.h_2D = makeHist()
        self.reader = ClueNtupleReader("v40", "cmssw", "data")
    
    def loadDataDataset(self):
        self.dataset_data = DRNDataset(self.reader, datasetComputationClass=self.datasetComputationClass, datasetType="full", simulation=False)

    def getPredictCallback(self):
        class FillHistogramCallback(pl.Callback):
            def __init__(self, h_2D:hist.Hist) -> None:
                super().__init__()
                self.h_2D = h_2D
            def on_predict_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
                self.h_2D.fill(batch.beamEnergy.detach().cpu().numpy(), pl_module.loss_params.mapNetworkOutputToEnergyEstimate(outputs, batch).detach().cpu().numpy())
        return FillHistogramCallback(self.h_2D)

    def runPredictionOnTrainer(self, trainer:pl.Trainer, ckpt_path:str):
        trainer.predict(model=self.model, dataloaders=DataLoader(self.dataset_data, batch_size=self.batch_size), ckpt_path=ckpt_path)
    
    def makeTrainerAndPredict(self, ckpt_path:str, trainer_kwargs:dict=dict(accelerator="gpu", devices=[3], logger=False)):
        if "callbacks" not in trainer_kwargs:
            trainer_kwargs["callbacks"] = []
        
        trainer_kwargs["callbacks"].append(self.getPredictCallback())
        trainer = pl.Trainer(**trainer_kwargs)

        self.runPredictionOnTrainer(trainer, ckpt_path)
        return self.h_2D

        
        

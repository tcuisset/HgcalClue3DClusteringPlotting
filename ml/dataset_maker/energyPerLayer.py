

import torch

from hists.dataframe import DataframeComputations

from ml.dataset_maker.tensors import SimpleTensorMakingComputation


class EnergyPerLayerTensorMaker(SimpleTensorMakingComputation):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs, neededBranches=["rechits_energy", "rechits_layer", "beamEnergy", "trueBeamEnergy"],
            rechits_columns=["rechits_energy", "rechits_layer"], tensorFileName="energyPerLayer")

    def computeTensor(self, comp:DataframeComputations) -> torch.Tensor:
        df = comp.rechits_totalReconstructedEnergyPerEventLayer.join(comp.trueBeamEnergy)
        return torch.tensor(df.rechits_energy_sum_perLayer.unstack().fillna(0).to_numpy()), torch.tensor(df.trueBeamEnergy.groupby("eventInternal").head(1).to_numpy())


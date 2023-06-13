import os
import typing

import torch
from torch_geometric.data import Data, InMemoryDataset
import numpy as np
import hist
import pandas as pd
import awkward as ak

from hists.parameters import beamEnergies
from hists.dataframe import DataframeComputations
from hists.custom_hists import beamEnergiesAxis, layerAxis_custom
from ntupleReaders.clue_ntuple_reader import ClueNtupleReader
from ntupleReaders.computation import BaseComputation, ComputationToolBase, computeAllFromTree
from ntupleReaders.tools import DataframeComputationsToolMaker

from ml.dataset_maker.tensors import TensorMakingComputation

neededBranchesGlobal = ["clus3D_energy", "clus3D_idxs", "clus3D_size",  'clus3D_x', 'clus3D_y', 'clus3D_z',
            "rechits_energy", "rechits_layer", "beamEnergy", "trueBeamEnergy", 'clus2D_x', 'clus2D_y', 'clus2D_z', 'clus2D_energy', 'clus2D_layer', 'clus2D_size', 'clus2D_rho', 'clus2D_delta', 'clus2D_pointType']


class RechitsTensorMaker(BaseComputation):
    shortName = "rechits"
    def __init__(self, beamEnergiesToSelect=beamEnergies, tensorFileName="rechitsGeometric", **kwargs) -> None:
        """ Parameters : 
         - beamEnergiesToSelect : beam energies to keep in dataset
        """
        self.requestedBranches = ["rechits_x", "rechits_y", "rechits_layer", "rechits_energy"]
        self.tensorFileName = tensorFileName
        self.beamEnergiesToSelect = beamEnergiesToSelect
        super().__init__(**kwargs, neededBranches=self.requestedBranches + ["clus3D_energy", "clus3D_idxs", "clus2D_idxs", "beamEnergy", "trueBeamEnergy"])
        
        self.geometric_data_objects = []

    def processBatch(self, array:ak.Array, report, computationTools:dict[typing.Type[ComputationToolBase], ComputationToolBase]) -> None:
        # remove events with no trackster
        array = array[ak.num(array.clus3D_energy, axis=-1) >= 1]
        # compute indices of 3D clusters for each event, sorted by energy
        # resulting array is event * [index] (with a single-element list)
        clus3D_energySortedIndices = ak.argsort(array['clus3D_energy'], axis=-1, ascending=False)[:, 0:1]

        # compute ids of 2D clusters that are in the main trackster
        # the [:, 0, :] removes the useless one-element list
        # resulting event * clus2D_ids
        clus2D_ids = array.clus3D_idxs[clus3D_energySortedIndices][:, 0, :]
        
        # compute the ids of all the rechits (flattening as we do not care about LC divisions)
        # event * rechit_id
        rechits_ids = ak.flatten(array.clus2D_idxs[clus2D_ids], axis=-1)
        

        columns = [array[colName][rechits_ids] for colName in self.requestedBranches]
        
        
        for eventBranches, beamEnergy, trueBeamEnergy in zip(zip(*columns), array.beamEnergy, array.trueBeamEnergy):
            if int(beamEnergy) in self.beamEnergiesToSelect:
                tensor_array = np.array(eventBranches, dtype=np.float32)
                tensor = torch.tensor(tensor_array.T)
                self.geometric_data_objects.append(Data(x=tensor, 
                    beamEnergy=torch.tensor(beamEnergy, dtype=torch.float32),
                    trueBeamEnergy=torch.tensor(trueBeamEnergy, dtype=torch.float32)))

    def saveTensor(self, pathToFolder:str) -> None:
        """ Save the tensor to file """
        try:
            tensorName = self.tensorFileName
        except AttributeError:
            tensorName = self.__class__.__name__
        torch.save(self.geometric_data_objects, os.path.join(pathToFolder, tensorName))
    
class LayerClustersTensorMaker(BaseComputation):
    shortName = "LC"
    def __init__(self, beamEnergiesToSelect=beamEnergies, tensorFileName="layerClustersGeometric", **kwargs) -> None:
        """ Parameters : 
         - beamEnergiesToSelect : beam energies to keep in dataset
        """
        self.requestedBranches = ["clus2D_x", "clus2D_y", "clus2D_layer", "clus2D_energy"]
        self.tensorFileName = tensorFileName
        self.beamEnergiesToSelect = beamEnergiesToSelect
        super().__init__(**kwargs, neededBranches=self.requestedBranches + ["clus3D_energy", "clus3D_idxs", "clus2D_idxs", "beamEnergy", "trueBeamEnergy"])
        
        self.geometric_data_objects = []

    def processBatch(self, array:ak.Array, report, computationTools:dict[typing.Type[ComputationToolBase], ComputationToolBase]) -> None:
        # remove events with no trackster
        array = array[ak.num(array.clus3D_energy, axis=-1) >= 1]
        # compute indices of 3D clusters for each event, sorted by energy
        # resulting array is event * [index] (with a single-element list)
        clus3D_energySortedIndices = ak.argsort(array['clus3D_energy'], axis=-1, ascending=False)[:, 0:1]

        # compute ids of 2D clusters that are in the main trackster
        # the [:, 0, :] removes the useless one-element list
        # resulting event * clus2D_ids
        clus2D_ids = array.clus3D_idxs[clus3D_energySortedIndices][:, 0, :]
        
        columns = [array[colName][clus2D_ids] for colName in self.requestedBranches]
        
        for eventBranches, beamEnergy, trueBeamEnergy in zip(zip(*columns), array.beamEnergy, array.trueBeamEnergy):
            if int(beamEnergy) in self.beamEnergiesToSelect:
                tensor_array = np.array(eventBranches, dtype=np.float32)
                tensor = torch.tensor(tensor_array.T)
                self.geometric_data_objects.append(Data(x=tensor, 
                    beamEnergy=torch.tensor(beamEnergy, dtype=torch.float32),
                    trueBeamEnergy=torch.tensor(trueBeamEnergy, dtype=torch.float32)))

    def saveTensor(self, pathToFolder:str) -> None:
        """ Save the tensor to file """
        try:
            tensorName = self.tensorFileName
        except AttributeError:
            tensorName = self.__class__.__name__
        torch.save(self.geometric_data_objects, os.path.join(pathToFolder, tensorName))
    


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
    optimalBatchSize = 1024
    """ Batch size to use to avoid out-of-memory on llrai01 """
    def __init__(self, beamEnergiesToSelect=beamEnergies, tensorFileName="rechitsGeometric", simulation=True, **kwargs) -> None:
        """ Parameters : 
         - beamEnergiesToSelect : beam energies to keep in dataset
        """
        self.simulation = simulation

        self.requestedBranches = ["rechits_x", "rechits_y", "rechits_layer", "rechits_energy"]
        neededBranches = self.requestedBranches + ["clus3D_energy", "clus3D_idxs", "clus2D_idxs", "beamEnergy"] # branches that need to be loaded from tree
        if simulation:
            neededBranches.append("trueBeamEnergy")
        self.tensorFileName = tensorFileName
        self.beamEnergiesToSelect = beamEnergiesToSelect
        super().__init__(**kwargs, neededBranches=neededBranches)
        
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
        

        # columns that are a tensor for each event
        perClusterColumns = [array[colName][rechits_ids] for colName in self.requestedBranches]

        # Columns that are a scalar per event (beamEnergy, gunEnergy)
        perEventColumns = [array.beamEnergy] # NB: beamEnery has to be the first
        perEventColumnNames = ["beamEnergy"]
        if self.simulation:
            perEventColumns.append(array.trueBeamEnergy)
            perEventColumnNames.append("trueBeamEnergy")

        for eventBranches, perEventValues in zip(zip(*perClusterColumns), zip(*perEventColumns)):
            beamEnergy = perEventValues[0]
            if int(beamEnergy) in self.beamEnergiesToSelect:
                tensor_array = np.array(eventBranches, dtype=np.float32)
                tensor = torch.tensor(tensor_array.T)

                self.geometric_data_objects.append(Data(
                    x=tensor, 
                    # append all scalar columns as scalar tensors
                    **{colName : torch.tensor(colValue, dtype=torch.float32) for colValue, colName in zip(perEventValues, perEventColumnNames, strict=True)}
                ))

    def saveTensor(self, pathToFolder:str) -> None:
        """ Save the tensor to file """
        try:
            tensorName = self.tensorFileName
        except AttributeError:
            tensorName = self.__class__.__name__
        torch.save(self.geometric_data_objects, os.path.join(pathToFolder, tensorName))
    
class LayerClustersTensorMaker(BaseComputation):
    shortName = "LC"
    optimalBatchSize = 2**14
    """ Batch size to use to avoid out-of-memory on llrai01 """
    def __init__(self, beamEnergiesToSelect=beamEnergies, tensorFileName="layerClustersGeometric", simulation=True, **kwargs) -> None:
        """ Parameters : 
         - beamEnergiesToSelect : beam energies to keep in dataset
         - simulation : if True (the default), will save gun energy. False is for data
        """
        self.simulation = simulation
        self.requestedBranches = ["clus2D_x", "clus2D_y", "clus2D_layer", "clus2D_energy"] # branches that will go in tensors
        neededBranches = self.requestedBranches + ["clus3D_energy", "clus3D_idxs", "clus2D_idxs", "beamEnergy"] # branches that need to be loaded from tree
        if simulation:
            neededBranches.append("trueBeamEnergy")

        self.tensorFileName = tensorFileName
        self.beamEnergiesToSelect = beamEnergiesToSelect
        super().__init__(**kwargs, neededBranches=neededBranches)
        
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
        
        # columns that are a tensor for each evebt
        perClusterColumns = [array[colName][clus2D_ids] for colName in self.requestedBranches]

        # Columns that are a scalar per event (beamEnergy, gunEnergy)
        perEventColumns = [array.beamEnergy] # NB: beamEnery has to be the first
        perEventColumnNames = ["beamEnergy"]
        if self.simulation:
            perEventColumns.append(array.trueBeamEnergy)
            perEventColumnNames.append("trueBeamEnergy")
        
        for eventBranches, perEventValues in zip(zip(*perClusterColumns), zip(*perEventColumns)):
            beamEnergy = perEventValues[0]
            if int(beamEnergy) in self.beamEnergiesToSelect:
                tensor_array = np.array(eventBranches, dtype=np.float32)
                tensor = torch.tensor(tensor_array.T)

                self.geometric_data_objects.append(Data(
                    x=tensor, 
                    # append all scalar columns as scalar tensors
                    **{colName : torch.tensor(colValue, dtype=torch.float32) for colName, colValue in zip(perEventColumnNames, perEventValues, strict=True)}
                ))

    def saveTensor(self, pathToFolder:str) -> None:
        """ Save the tensor to file """
        try:
            tensorName = self.tensorFileName
        except AttributeError:
            tensorName = self.__class__.__name__
        torch.save(self.geometric_data_objects, os.path.join(pathToFolder, tensorName))
    


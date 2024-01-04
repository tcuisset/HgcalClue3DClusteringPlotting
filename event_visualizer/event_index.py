import collections
import itertools
import functools
from functools import cached_property
import weakref

import uproot
import awkward as ak
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import hists.parameters

# Uniquely identify an event. For data, beamEnergy is not necessary and can be set to None.
# For simulation, it is needed since ntupleNumber overlap for different beam energies
EventID = collections.namedtuple("EventID", ["beamEnergy", "ntupleNumber", "event"])



def memoized_method(*lru_args, **lru_kwargs):
    """Small code taken from https://stackoverflow.com/a/33672499 
    So we can cache method results with cache that is per instance rather than per class
    as using @functools.cache leads to memery leaks when used on an instance method"""
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(self, *args, **kwargs):
            # We're storing the wrapped method inside the instance. If we had
            # a strong reference to self the instance would never die.
            self_weak = weakref.ref(self)
            @functools.wraps(func)
            @functools.lru_cache(*lru_args, **lru_kwargs)
            def cached_method(*args, **kwargs):
                return func(self_weak(), *args, **kwargs)
            setattr(self, func.__name__, cached_method)
            return cached_method(*args, **kwargs)
        return wrapped_func
    return decorator

class DTypeFinder:
    """ Find the smallest dtype for branches of tree """
    def __init__(self, event_array_dict:dict[str, np.ndarray]) -> None:
        self.event_array_dict = event_array_dict
    
    @memoized_method(maxsize=None)
    def bestDtypeFor(self, column:str) -> np.dtype:
        if column == "index":
            return column, np.min_scalar_type(len(self.event_array_dict["beamEnergy"]))
        else:
            return column, np.min_scalar_type(np.max(self.event_array_dict[column]))

def findDtypeInfo(dtype:np.dtype):
    """ Get the information for the dtype, that works for both int and float types. You can call min or max on the resulting object """
    try:
        return np.iinfo(dtype)
    except ValueError:
        return np.finfo(dtype)

class EventIndex:
    """ Index of events """
    def __init__(self, event_array_dict:dict[str, np.ndarray], columns:list[str]) -> None:
        """ 
         - event_array_dict : dictionnary column name -> numpy array
         - columns: list of columns, order of which to sort on
        """
        self.event_array_dict = event_array_dict
        
        dtypeFinder = DTypeFinder(event_array_dict)
        self._index_structured_dtype = np.dtype([dtypeFinder.bestDtypeFor(column) for column in (columns + ["index"])])

        index_array = np.arange(0, len(event_array_dict["event"]), dtype=self._index_structured_dtype["index"])
        self.record_array = np.core.records.fromarrays(list(event_array_dict.values())+[index_array], dtype=self._index_structured_dtype)
        self.record_array.sort()

    def findRecord(self, columnValues:list):
        """ Find a record, by the values of columns. The values in columnValues should match the order of columns given in __init__ 
        Returns a numpy record with columns values and a record called "index" holding index in original array
        """
        value = np.array(tuple(columnValues + [0]), dtype=self._index_structured_dtype)
        foundRecord = self.record_array[np.searchsorted(self.record_array, value)]
        return foundRecord
    
    def _makeRecordFromColumnValues(self, firstColumnValues, min=True):
        value_list = []
        for (dtype, _), value in itertools.zip_longest(self._index_structured_dtype.fields.values(), firstColumnValues):
            if value is None:
                if min:
                    value_list.append(findDtypeInfo(dtype).min)
                else:
                    value_list.append(findDtypeInfo(dtype).max)
            else:
                value_list.append(value)
        return np.array(tuple(value_list), dtype=self._index_structured_dtype)
    
    def findRange(self, firstColumnValues:list) -> np.ndarray:
        """ Returns a view over the range of records selected by column values given in firstColumnValues.
        This should be a continuous subset from the beginning of columns, in same order.
        """
        record_lookup_begin = self._makeRecordFromColumnValues(firstColumnValues, min=True)
        record_lookup_end = self._makeRecordFromColumnValues(firstColumnValues, min=False)
        
        indexMin, indexMax = np.searchsorted(self.record_array, record_lookup_begin, side="left"), np.searchsorted(self.record_array, record_lookup_end, side="right")
        #return indexMin, indexMax
        return self.record_array[indexMin:indexMax]


class EventLoader:
    def __init__(self, pathToFile:str, datatype:str=None) -> None:
        """ pathToFile is path to CLUE_clusters.root file (included)
        datatype can be "data", "simulation", a simulation tag ("sim_proton_v46_patchMIP"), None (unknown) """
        self.file = uproot.open(pathToFile, object_cache=None, array_cache=None)
        self.tree:uproot.TTree = self.file["clusters"]
        self.eventIndex = EventIndex(self.tree.arrays(filter_name=["beamEnergy", "ntupleNumber", "event"], library="np"),
                                        columns=["beamEnergy", "ntupleNumber", "event"])
        self.datatype = datatype

    @cached_property
    def clueParameters(self):
        return self.file["clueParams"].members
    
    @cached_property
    def clue3DParameters(self):
        return self.file["clue3DParams"].members

    def _locateEvent(self, eventId:EventID):
        """ Compute the index in the tree of the given event, using the event index """
        if eventId.beamEnergy is None:
            #eventId = eventId._replace(beamEnergy=0)
            raise ValueError("For now, you must specify beamEnergy")
        try:
            foundRecord = self.eventIndex.findRecord([eventId.beamEnergy, eventId.ntupleNumber, eventId.event])
            #foundEvent = self.eventIndex[foundRecord.index]
            if (foundRecord["event"] == eventId.event and foundRecord["ntupleNumber"] == eventId.ntupleNumber 
                and (eventId.beamEnergy == 0 or foundRecord["beamEnergy"] == eventId.beamEnergy)) :
                return foundRecord["index"]
        except IndexError:
            pass
        return None

    def loadEvent(self, eventId:EventID) -> "LoadedEvent": # (forward reference to LoadedEvent)
        eventLocation = self._locateEvent(eventId)
        if eventLocation is None:
            raise RuntimeError("Could not find event")
        eventList = self.tree.arrays(entry_start=eventLocation, entry_stop=eventLocation+1)
        if len(eventList) > 1:
            raise RuntimeError("Uproot problem")
            #print("WARNING : duplicate event numbers") # should not happen 
        if len(eventList) < 1:
            raise RuntimeError("Could not find event") 
        return LoadedEvent(eventList[0], self)
    
    @cached_property
    def ntuplesEnergies(self) -> ak.Array:
        """ Gets a list of records with the unique ntupleNumber and beamEnergy pairs"""
        return np.unique(self.tree.arrays(["ntupleNumber", "beamEnergy"]))

    def eventNumbersPerNtuple(self, beamEnergy, ntupleNumber) -> np.ndarray:
        return self.eventIndex.findRange([beamEnergy, ntupleNumber]).event

class LoadedEvent:
    def __init__(self, record:ak.Record, el:EventLoader) -> None:
        self.record:ak.Record = ak.to_packed(record) # Pack the event so reduces greatly the pickled size
        self.clueParameters = el.clueParameters
        self.clue3DParameters = el.clueParameters
        self.datatype = el.datatype
    
    @property
    def clus3D_df(self):
        return (ak.to_dataframe(self.record[
            ["clus3D_x", "clus3D_y", "clus3D_z", "clus3D_energy", "clus3D_size"]
            ], 
            levelname=lambda i : {0:"clus3D_id"}[i])
        ).reset_index().set_index("clus3D_id") # Transform a MultiIndex with one level to a regular index
    
    def clus3D_ids(self, sortDecreasingEnergy=False) -> list[int]:
        """ Get list of 3D clusters ids in current event (not including NaN)"""
        df = self.clus3D_df
        if sortDecreasingEnergy:
            df = df.sort_values("clus3D_energy", ascending=False)
        return df.index.get_level_values("clus3D_id").drop_duplicates().to_list()

    #Note that in the dataframes event is not the same as the "event number" in the ntuples
    @cached_property
    def clus2D_df(self):
        clusters3D_withClus2Did = (ak.to_dataframe(
                self.record[["clus3D_idxs"]],
                levelname=lambda i : {0:"clus3D_id", 1:"clus2D_internal_id"}[i]
            )
            # clus2D_internal_id is an identifier counting 2D clusters in each 3D cluster (it is NOT the same as clus2D_id, which is unique per event, whilst clus2D_internal_id is only unique per 3D cluster)
            .reset_index(level="clus2D_internal_id", drop=True)
            .rename(columns={"clus3D_idxs" : "clus2D_id"})
            .reset_index()
        )
        clus3D_merged = (
            pd.merge(
                (ak.to_dataframe(self.record[["clus2D_x", "clus2D_y", "clus2D_z", "clus2D_energy", "clus2D_layer", "clus2D_size",
                    "clus2D_rho", "clus2D_delta", "clus2D_pointType", "clus2D_nearestHigher"]], 
                        levelname=lambda i : {0:"clus2D_id"}[i])
                    .reset_index()
                ),
                clusters3D_withClus2Did,
                on="clus2D_id",
                how="left"
            )
            .set_index("clus2D_id")
            .pipe(makeCumulativeEnergy, prefix="clus2D")
        )
        return clus3D_merged.join(clus3D_merged[["clus2D_x", "clus2D_y", "clus2D_z", "clus2D_layer", "clus2D_energy"]], on="clus2D_nearestHigher", rsuffix="_ofNearestHigher")

    @property
    def clus2D_ids(self) -> list[int]:
        """ Get list of 2D clusters ids in current event (not including NaN)"""
        return self.clus2D_df.index.get_level_values("clus2D_id").drop_duplicates().to_list()

    @cached_property
    def rechits_df(self) -> pd.DataFrame:
        """ Indexed by rechit_id. Left-joined to clusters2D and clusters3D (NaN if outlier) """
        clusters2D_with_rechit_id = (
            ak.to_dataframe(
                self.record[[
                    "clus2D_idxs", "clus2D_pointType"
                ]], 
                levelname=lambda i : {0:"clus2D_id", 1:"rechit_internal_id"}[i]
            )
            .reset_index(level="rechit_internal_id", drop=True)
            .rename(columns={"clus2D_idxs" : "rechits_id"})
            .reset_index()
        )
        clus2D_merged = pd.merge(
            ak.to_dataframe(self.record[
                ["rechits_x", "rechits_y", "rechits_z", "rechits_energy", "rechits_layer",
                "rechits_rho", "rechits_delta", "rechits_nearestHigher", "rechits_pointType"]], 
                levelname=lambda i : {0:"rechits_id"}[i]
            ).reset_index(),
            clusters2D_with_rechit_id,
            on="rechits_id",
            how="left",
        )#.set_index("rechits_id")

        clusters3D_withClus2DId = (
            ak.to_dataframe(
                self.record[["clus3D_idxs"]],
                levelname=lambda i : {0:"clus3D_id", 1:"clus2D_internal_id"}[i]
            )
            # clus2D_internal_id is an identifier counting 2D clusters in each 3D cluster (it is NOT the same as clus2D_id, which is unique per event, whilst clus2D_internal_id is only unique per 3D cluster)
            .reset_index(level="clus2D_internal_id", drop=True)
            .rename(columns={"clus3D_idxs" : "clus2D_id"})
            .reset_index()
        )
        final_df = (
            pd.merge(
                clus2D_merged,
                clusters3D_withClus2DId,
                on="clus2D_id",
                how="left"
            )
            .set_index("rechits_id")
            .pipe(makeCumulativeEnergy, prefix="rechits")
        )
        return final_df.join(final_df[["rechits_x", "rechits_y", "rechits_z", "rechits_layer", "rechits_energy",]], on=["rechits_nearestHigher"], rsuffix="_ofNearestHigher")

    @property
    def impact_df(self) -> pd.DataFrame:
        """ Index : event (always 0)
        Columns : layer impact_x impact_y impact_z (impact_z is mapped from layer)
        Only rows which have a rechit in the event are kept"""
        df = ak.to_dataframe(self.record[["impactX", "impactY"]],
            levelname=lambda i : {0:"layer_minus_one"}[i]).reset_index().rename({"impactX" : "impact_x", "impactY": "impact_y"}, axis="columns")
        df["impact_layer"] = df["layer_minus_one"] + 1
        df = df.drop("layer_minus_one", axis="columns")

        return df.assign(impact_z=df.impact_layer.map(hists.parameters.layerToZMapping)).dropna()

def makeCumulativeEnergy(df:pd.DataFrame, prefix:str):
    """ df must be indexed by clus2D_id or rechits_id"""
    new_df = df.copy()
    new_df[f"{prefix}_cumulativeEnergy"] = 0.
    cumulEnergy = new_df[f"{prefix}_cumulativeEnergy"].copy()

    for row in df.itertuples(index=True):
        current_clus2D_id = row.Index

        while new_df[f"{prefix}_nearestHigher"].loc[current_clus2D_id] != -1:
            cumulEnergy.loc[current_clus2D_id] += getattr(row, f"{prefix}_energy")
            current_clus2D_id = new_df[f"{prefix}_nearestHigher"].loc[current_clus2D_id]
        
    new_df[f"{prefix}_cumulativeEnergy"] = cumulEnergy
    return new_df

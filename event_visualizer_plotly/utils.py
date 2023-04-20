import math
import collections
from functools import cached_property

import uproot
import awkward as ak
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import hists.parameters

EventID = collections.namedtuple("EventID", ["ntupleNumber", "event"])

class EventLoader:
    def __init__(self, pathToFile:str) -> None:
        """ pathToFile is path to CLUE_clusters.root file (included)"""
        self.file = uproot.open(pathToFile)
        self.tree:uproot.TTree = self.file["clusters"]

    @cached_property
    def clueParameters(self):
        return self.file["clueParams"].members
    
    @cached_property
    def clue3DParameters(self):
        return self.file["clue3DParams"].members

    def _locateEvent(self, eventId:EventID) -> uproot.behaviors.TBranch.Report:
        for array, report in self.tree.iterate(["ntupleNumber", "event"], report=True, library="np",
            cut=f"(ntupleNumber == {eventId.ntupleNumber}) & (event=={eventId.event})", step_size=1000):
            if len(array["ntupleNumber"]) > 0:
                return report
        return None

    def loadEvent(self, eventId:EventID) -> "LoadedEvent": # (forward reference to LoadedEvent)
        eventLocation = self._locateEvent(eventId)
        if eventLocation is None:
            raise RuntimeError("Could not find event")
        eventList = self.tree.arrays(cut=f"(ntupleNumber == {eventId.ntupleNumber}) & (event=={eventId.event})",
            entry_start=eventLocation.tree_entry_start, entry_stop=eventLocation.tree_entry_stop)
        if len(eventList) > 1:
            print("WARNING : duplicate event numbers")
        if len(eventList) < 1:
            raise RuntimeError("Could not find event") 
        return LoadedEvent(eventList[0], self)
    
    def getNtuplesEnergies(self) -> ak.Array:
        """ Gets a list of records with the unique ntupleNumber and beamEnergy pairs"""
        return np.unique(self.tree.arrays(["ntupleNumber", "beamEnergy"]))

class LoadedEvent:
    def __init__(self, record:ak.Record, el:EventLoader) -> None:
        self.record:ak.Record = record
        self.el:EventLoader = el

    @property
    def clueParameters(self):
        return self.el.clueParameters
    
    @property
    def clue3DParameters(self):
        return self.el.clue3DParameters



def create3DFigure(title:str) -> go.Figure:
    fig = go.Figure(
        layout=go.Layout(
            title=go.layout.Title(text=title),
            #width=1200,
            #height=600,
            autosize=True,
            dragmode="orbit"
        )
    )
    camera = dict(
        eye=dict(x=0, y=0., z=-2.5),
        up=dict(x=0, y=0, z=1),
    )
    fig.update_layout(scene_camera=camera)
    return fig


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

def getPointTypeStringForRechits(clus2D_id:float, grouped_df:pd.DataFrame):
    """ 
    Parameters : 
    - clus2D_id : the 2D cluster id of the grouped_df (can be NaN in case not in a layer cluster or in a masked layer cluster)
    - grouped_df : rechitds_df (must have rechits_pointType)"""
    pointTypeString = []
    pointTypeDict = {0:"Follower", 1:"Seed", 2:"Outlier"}
    for row in grouped_df.itertuples():
        if row.rechits_pointType == 1 and math.isnan(clus2D_id): # Seed, but clus2D_id is NaN
            pointTypeString.append("Masked cluster seed")
        else:
            pointTypeString.append(pointTypeDict[row.rechits_pointType])
    return pointTypeString

class BaseVisualization:
    def __init__(self, event:LoadedEvent) -> None:
        #self.fig.update_layout(legend=dict(groupclick="togglegroup"))
        self.event:LoadedEvent = event

    @cached_property
    def clus3D_df(self):
        return (ak.to_dataframe(self.event.record[
            ["clus3D_x", "clus3D_y", "clus3D_z", "clus3D_energy", "clus3D_size"]
            ], 
            levelname=lambda i : {0:"clus3D_id"}[i])
        ).reset_index().set_index("clus3D_id") # Transform a MultiIndex with one level to a regular index
    
    #Note that in the dataframes event is not the same as the "event number" in the ntuples
    @cached_property
    def clus2D_df(self):
        clusters3D_withClus2Did = (ak.to_dataframe(
                self.event.record[["clus3D_idxs"]],
                levelname=lambda i : {0:"clus3D_id", 1:"clus2D_internal_id"}[i]
            )
            # clus2D_internal_id is an identifier counting 2D clusters in each 3D cluster (it is NOT the same as clus2D_id, which is unique per event, whilst clus2D_internal_id is only unique per 3D cluster)
            .reset_index(level="clus2D_internal_id", drop=True)
            .rename(columns={"clus3D_idxs" : "clus2D_id"})
            .reset_index()
        )
        clus3D_merged = (
            pd.merge(
                (ak.to_dataframe(self.event.record[["clus2D_x", "clus2D_y", "clus2D_z", "clus2D_energy", "clus2D_layer", "clus2D_size",
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
        return clus3D_merged.join(clus3D_merged[["clus2D_x", "clus2D_y", "clus2D_z"]], on="clus2D_nearestHigher", rsuffix="_ofNearestHigher")


    @cached_property
    def rechits_df(self) -> pd.DataFrame:
        """ Indexed by rechit_id. Left-joined to clusters2D and clusters3D (NaN if outlier) """
        clusters2D_with_rechit_id = (
            ak.to_dataframe(
                self.event.record[[
                    "clus2D_idxs", "clus2D_pointType"
                ]], 
                levelname=lambda i : {0:"clus2D_id", 1:"rechit_internal_id"}[i]
            )
            .reset_index(level="rechit_internal_id", drop=True)
            .rename(columns={"clus2D_idxs" : "rechits_id"})
            .reset_index()
        )
        clus2D_merged = pd.merge(
            ak.to_dataframe(self.event.record[
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
                self.event.record[["clus3D_idxs"]],
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
        return final_df.join(final_df[["rechits_x", "rechits_y", "rechits_z"]], on=["rechits_nearestHigher"], rsuffix="_ofNearestHigher")

    @property
    def impact_df(self) -> pd.DataFrame:
        """ Index : event (always 0)
        Columns : layer impactX impactY impactZ (impactZ is mapped from layer)
        Only rows which have a rechit in the event are kept"""
        df = ak.to_dataframe(self.event.record[["impactX", "impactY"]],
            levelname=lambda i : {0:"layer_minus_one"}[i]).reset_index()
        df["layer"] = df["layer_minus_one"] + 1
        df = df.drop("layer_minus_one", axis="columns")

        return df.assign(impactZ=df.layer.map(hists.parameters.layerToZMapping)).dropna()

def makeArrow3D(x1, x2, y1, y2, z1, z2, dictLine=dict(), dictCone=dict(), color="blue"):
    traces = []
    try:
        lengthFactor = 1./math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        traces.append(go.Scatter3d(
            **collections.ChainMap(dictLine, dict( # Give preference to keywords from arguments over those specified here
                mode="lines",
                hoverinfo='skip',
                x=[x1, x2],
                y=[y1, y2],
                z=[z1, z2],
                marker_color=color,
            ))
        ))
        traces.append(go.Cone(
            **collections.ChainMap(dictCone, dict(
                x=[x2], y=[y2], z=[z2],
                u=[lengthFactor*(x2-x1)],
                v=[lengthFactor*(y2-y1)],
                w=[lengthFactor*(z2-z1)],
                sizeref=0.1,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                anchor="tip",
                hoverinfo="skip"
            ))
        ))
    except ZeroDivisionError: # In case two rechits are in the same point (happens for calibration pads)
        pass
    return traces


def NaNColorMap(d:dict[float, str], NaNColor:str):
    def mapFct(val:float) -> str:
        if math.isnan(val):
            return NaNColor
        else:
            return d[val]
    return mapFct


class MarkerSizeLinearScaler:
    def __init__(self, allEnergiesSeries:pd.Series, maxMarkerSize=10) -> None:
        self.maxEnergy = allEnergiesSeries.max()
        self.maxMarkerSize = maxMarkerSize
    
    def scale(self, series:pd.Series):
        return (series / self.maxEnergy * self.maxMarkerSize).clip(lower=1)

class MarkerSizeLogScaler:
    def __init__(self, allEnergiesSeries:pd.Series, maxMarkerSize=10, minMarkerSize=1) -> None:
        """ Log scale such that min(allEnergiesSeries) maps to minMarkerSize, and max(allEnergiesSeries) maps to maxMarkerSize
        Write size = b * ln(E/a) """
        minEnergy = allEnergiesSeries.min()
        maxEnergy = allEnergiesSeries.max()
        if minEnergy < maxEnergy:
            self._ln_a = (maxMarkerSize * math.log(minEnergy) - minMarkerSize*math.log(maxEnergy)) / (maxMarkerSize - minMarkerSize)
            self._b = minMarkerSize / (math.log(minEnergy) - self._ln_a)
        else:
            # Deal with the case with only one energy
            self._ln_a = math.log(minEnergy) - 1
            self._b = maxMarkerSize # Put desired marker size here. For now just take the max

    def scale(self, series:pd.Series):
        return (self._b * (np.log(series) - self._ln_a)).clip(lower=1)
    

    
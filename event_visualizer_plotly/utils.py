import math
import collections

import uproot
import plotly.graph_objects as go

from hists.dataframe import *

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

    def locateEvent(self, eventId:EventID) -> uproot.behaviors.TBranch.Report:
        for array, report in self.tree.iterate(["ntupleNumber", "event"], report=True, library="np",
            cut=f"(ntupleNumber == {eventId.ntupleNumber}) & (event=={eventId.event})", step_size=1000):
            if len(array["ntupleNumber"]) > 0:
                return report
        return None

    def loadEvent(self, eventId:EventID):# -> LoadedEvent:
        eventLocation = self.locateEvent(eventId)
        if eventLocation is None:
            raise RuntimeError("Could not find event")
        eventList = self.tree.arrays(cut=f"(ntupleNumber == {eventId.ntupleNumber}) & (event=={eventId.event})",
            entry_start=eventLocation.tree_entry_start, entry_stop=eventLocation.tree_entry_stop)
        if len(eventList) > 1:
            print("WARNING : duplicate event numbers")
        if len(eventList) < 1:
            raise RuntimeError("Could not find event") 
        return LoadedEvent(eventList[0], self)

class LoadedEvent:
    def __init__(self, record:ak.Record, el:EventLoader) -> None:
        self.comp = DataframeComputations(ak.Array([record]))
        self.el = el

    @property
    def clueParameters(self):
        return self.el.clueParameters
    
    @property
    def clue3DParameters(self):
        return self.el.clue3DParameters



def create3DFigure() -> go.Figure:
    fig = go.Figure(
        layout=go.Layout(
            title=go.layout.Title(text="Event visualizer"),
            #width=1200,
            #height=600,
            autosize=True
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

class BaseVisualization:
    def __init__(self, event:LoadedEvent) -> None:
        #self.fig.update_layout(legend=dict(groupclick="togglegroup"))
        self.event = event

    @cached_property
    def clus3D_df(self):
        return self.event.comp.clusters3D.loc[0]
    
    #Note that in the dataframes event is not the same as the "event number" in the ntuples
    @cached_property
    def clus2D_df(self):
        df_clus2D = (self.event.comp
            .clusters3D_merged_2D_custom(self.event.comp.clusters3D_with_clus2D_id, self.event.comp.clusters2D_withNearestHigher)
            .set_index(["event", "clus3D_id", "clus2D_id"])
            .loc[0]
        )
        return (df_clus2D
            .join(df_clus2D[["clus2D_x", "clus2D_y", "clus2D_z"]], on=["clus3D_id", "clus2D_nearestHigher"], rsuffix="_ofNearestHigher")
            .reset_index(level="clus3D_id")
            .pipe(makeCumulativeEnergy, prefix="clus2D")
            .reset_index()
            .set_index(["clus3D_id", "clus2D_id"])
        )


    @cached_property
    def rechits_df(self):
        df_rechits = (self.event.comp
            .clusters3D_merged_rechits_custom(["rechits_x", "rechits_y", "rechits_z", "rechits_energy", 
                "rechits_rho", "rechits_delta", "rechits_nearestHigher", "rechits_pointType"])
            .loc[0]

            .reset_index(level=["clus3D_id", "rechits_layer"])
            .pipe(makeCumulativeEnergy, prefix="rechits")
            .reset_index()
            .set_index(["clus3D_id", "clus2D_id"])
        )
        return df_rechits.join(df_rechits[["rechits_id", "rechits_x", "rechits_y", "rechits_z"]].set_index("rechits_id"), on=["rechits_nearestHigher"], rsuffix="_ofNearestHigher")
    
    @property
    def impact_df(self) -> pd.DataFrame:
        """ Index : event (always 0)
        Columns : layer impactX impactY impactZ (impactZ is mapped from layer)
        Only rows which have a rechit in the event are kept"""
        df = self.event.comp.impact.reset_index(level="layer")
        return df.assign(impactZ=df.layer.map(self.event.comp.layerToZMapping)).dropna()

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



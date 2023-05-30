import math
import collections
import itertools

import awkward as ak
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from event_visualizer.event_index import LoadedEvent        

def create3DFigure(title:str) -> go.Figure:
    fig = go.Figure(
        layout=go.Layout(
            title=go.layout.Title(text=title),
            #width=1200,
            #height=600,
            autosize=True,
            dragmode="orbit",
            scene=dict(
                aspectratio=dict(x=1., y=1., z=3.),
                camera = dict(
                    eye=dict(x=0, y=0., z=-2.5),
                    up=dict(x=0, y=0, z=1),
                ),
            ),
        )
    )
    return fig


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

        # Symbols for 2D clusters, all 2D clusters member of a given 3D cluster get the same symbol
        self.clus3D_symbols_3Dview = itertools.cycle(['diamond', 'square', 'circle', 'cross', 'x'])
        self.clus3D_symbols_2Dview = itertools.cycle(["diamond", "square", "cross",  "pentagon", "star", "star-triangle-up", "star-square", "hourglass", "hexagram", "star-diamond", "circle-cross", "diamond-tall", "square-cross"])
        # Symbols for 2D clusters that are (followers of) an outlier in CLUE3D
        self.clus3D_symbols_outlier_3Dview = itertools.cycle([ 'circle-open', 'square-open', 'diamond-open'])
        self.clus3D_symbols_outlier_2Dview = itertools.cycle([ "cross-open-dot", "pentagon-open-dot", "star-open-dot", "star-square-open-dot", "diamond-open-dot", "heaxagram-open-dot", "diamond-tall-open-dot", "diamond-wide-open-dot", "hash-open-dot"])
        
        self.mapClus3Did_symbol_3Dview = {clus3D_id : next(self.clus3D_symbols_3Dview) for clus3D_id in self.event.clus3D_ids(sortDecreasingEnergy=True)}
        self.mapClus3Did_symbol_2Dview = {clus3D_id : next(self.clus3D_symbols_2Dview) for clus3D_id in self.event.clus3D_ids(sortDecreasingEnergy=True)}

def makeArrow3D(x1, x2, y1, y2, z1, z2, dictLine=dict(), dictCone=dict(), dictCombined=dict(), color="blue", sizeFactor=1):
    """ Draw an arrow from x1, y1, z1 to x1, y2, z2
    Parameters : 
     - dictLine : dict of kwargs passed to Scatter3D
     - dictCone : dict of kwargs passed to Cone
     - dictCombined : dict of kwargs passed to both
     - color : color of arrow
     - sizeFactor : multiplicative factor on cone size
    """
    traces = []
    try:
        lengthFactor = sizeFactor/math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        # Use collections.ChainMap to merge dictionnaries in sequence
        # Give preference to keywords from arguments over those specified here
        traces.append(go.Scatter3d(
            **collections.ChainMap(dictLine, dictCombined, dict( 
                mode="lines",
                hoverinfo='skip',
                x=[x1, x2],
                y=[y1, y2],
                z=[z1, z2],
                marker_color=color,
            ))
        ))
        traces.append(go.Cone(
            **collections.ChainMap(dictCone, dictCombined, dict(
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

    def scale(self, val:pd.Series): # can be pd.Series or float
        if isinstance(val, pd.Series):
            return (self._b * (np.log(val) - self._ln_a)).clip(lower=1)
        else:
            return max(1, self._b * (np.log(val) - self._ln_a))
    

def makeCylinderCoordinates(r, h, axisX=0, axisY=0, z0=0, nt=100, nv =50):
    """
    parametrize the cylinder of axis (z), of radius r, height h, base point z coordinate z0
    axisX and axisY are the x and y coordinates of the axis
    Returns x, y, z points coordinates
    """
    theta = np.linspace(0, 2*np.pi, nt)
    v = np.linspace(z0, z0+h, nv )
    theta, v = np.meshgrid(theta, v)
    x = r*np.cos(theta) + axisX
    y = r*np.sin(theta) + axisY
    z = v
    return x, y, z
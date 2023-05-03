import random
import pandas as pd

from event_visualizer_plotly.vis_layer import LayerVisualization
from event_visualizer_plotly.vis_clue3D import Clue3DVisualization
from event_visualizer_plotly.utils import EventLoader, EventID, LoadedEvent
from event_visualizer_plotly.locateEvents.utils import makeDashLink, makeCsvRow, printCsvRowsFromDf
from event_visualizer_plotly.locateEvents.sample_chooser import EventDisplayList


def loadEventFromDf_iloc(df, index, eventLoader) -> tuple[LoadedEvent, int]:
    """ Returns LoadedEvent, layerNb tuple. index is integer index starting from 0 into df (uses iloc) """
    return eventLoader.loadEvent(EventID(df.ntupleNumber.iloc[index], df.event.iloc[index])), df.clus2D_layer.iloc[index]

def loadEventFromDf_loc(df, index, eventLoader) -> tuple[LoadedEvent, int]:
    """ Returns LoadedEvent, layerNb tuple. index is eventInternal """
    return eventLoader.loadEvent(EventID(df.ntupleNumber.loc[index], df.event.loc[index])), df.clus2D_layer.loc[index]


def plotEventLayer(event, layer):
    vis_layer = (LayerVisualization(event, layerNb=layer) 
        .add2DClusters()
        .addRechits()
        .addImpactPoint()
        .addCircleSearchForComputingClusterPosition()
        #.addDetectorExtent()
        )
    vis_layer.fig.show()

def plotEvent3D(event, _):
    vis_3d = (Clue3DVisualization(event)
        .add3DClusters()
        .add2DClusters()
        .addRechits(hiddenByDefault=True)
        .addImpactTrajectory()
        .addDetectorCylinder()
        .addSliders()
    )
    vis_3d.fig.show()

class EventDisplay:
    def __init__(self, df:pd.DataFrame, eventLoader:EventLoader) -> None:
        self.df = df
        self.el = eventLoader
        self.shuffledIndex = None
    
    def showEvent(self, index, method=loadEventFromDf_iloc):
        event_layer = method(self.df, index, self.el)
        plotEventLayer(*event_layer)
        plotEvent3D(*event_layer)
    
    def sampleRandom(self):
        if self.shuffledIndex is None:
            self.shuffledIndex = self.df.index.to_list()
            random.shuffle(self.shuffledIndex)
            self.shuffledIndexIter = iter(self.shuffledIndex)
        
        index = next(self.shuffledIndexIter)
        self.showEvent(index, method=loadEventFromDf_loc)
import bokeh.models

from HistogramLib.bokeh import *
from HistogramLib.projection_manager import HistKindSelector
from HistogramLib.histogram import HistogramKind

beamEnergies = [20, 50, 80, 100, 120, 150, 200, 250, 300]
datatypes = ['data', 'sim_proton', 'sim_noproton']

def makeLayerSelector():
    return RangeAxisSelector("layer",
        title="Layer selection",
        start=0,
        end=30,
        step=1,
        value=(1,28)
    )

def makeBeamEnergySelector():
    return MultiSelectAxisSelector("beamEnergy",
        title="Beam energy",
        options=[(str(value), str(value) + " GeV") for value in beamEnergies],
        value=[str(energy) for energy in beamEnergies],
        height=200
    )

def makeMainOrAllTrackstersSelector():
    """ For CLUE3D, whether we use all 3D clusters per event or only the cluster with the highest energy """
    return RadioButtonGroupAxisSelector("mainOrAllTracksters",
        name="mainOrAllTracksters",
        labels=["allTracksters", "mainTrackster"],
        active=0
    )

def makeCluster3DSizeSelector():
    return SliderMinWithOverflowAxisSelector("clus3D_size",
        title="Min 3D cluster size (ie nb of 2D clusters)",
        start=1,
        end=10,
        value=1
    )

class DatatypeSelector(HistogramIdSelector):
    def __init__(self) -> None:
        self.labels = ["data", "sim_proton", "sim_noproton"]
        self.widget = bokeh.models.RadioButtonGroup(
            name="datatype",
            labels=self.labels,
            active=0
        )

    def fillHistId(self, shelfId:ShelfId):
        # self.radio.value is the button number that is pressed -> map it to label
        shelfId.datatype = self.labels[self.widget.active]
    
    def registerCallback(self, callback):
        self.widget.on_change('active', callback)

class PlaceholderDatatypeSelector(HistogramIdSelector):
    def fillHistId(self, shelfId:ShelfId):
        # self.radio.value is the button number that is pressed -> map it to label
        shelfId.datatype = "data"
    def registerCallback(self, callback):
        pass

class ClueParamsSelector(HistogramIdSelector):
    def __init__(self, clueParamList) -> None:
        self.widget = bokeh.models.RadioButtonGroup(
            name="clue_params",
            labels=clueParamList,
            active=0
        )
    
    def fillHistId(self, shelfId:ShelfId):
        shelfId.clueParamName = self.widget.labels[self.widget.active]

    def registerCallback(self, callback):
        self.widget.on_change('active', callback)

class PlaceholderClueParamsSelector(HistogramIdSelector):
    def fillHistId(self, shelfId:ShelfId):
        shelfId.clueParamName = "default"

class HistogramKindRadioButton(HistKindSelector):
    labels_dict = {"Count" : HistogramKind.COUNT,
                    "Weight" : HistogramKind.WEIGHT,
                    "Profile" : HistogramKind.PROFILE}

    def __init__(self) -> None:
        self.widget = bokeh.models.RadioButtonGroup(
            labels=list(self.labels_dict.keys()),
            active=0
        )

    def registerCallback(self, callback):
        self.widget.on_change("active", callback)
    
    def getSelection(self) -> HistogramKind:
        return self.labels_dict[self.widget.labels[self.widget.active]]

import argparse

def parseArgs():
    parser = argparse.ArgumentParser(description="Plotting code to be run using Bokeh server, use bokeh serve SCRIPT.py --args ARGS")
    parser.add_argument("--hist-folder", dest="hist_folder",
        help="path to folder holding all histograms. Will load recursively all clueparams and datatypes inside this folder")
    parser.add_argument("--single-file", dest='single_file',
        help="Only load a single pickle file (for debugging), given by this full path to the file")
    return parser.parse_args()

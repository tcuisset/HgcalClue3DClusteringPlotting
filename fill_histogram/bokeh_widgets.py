import bokeh.models

from HistogramLib.bokeh import *

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


class DatatypeSelector(ShelfIdSelector):
    def __init__(self) -> None:
        self.labels = ["data", "sim_proton", "sim_noproton"]
        self.widget = bokeh.models.RadioButtonGroup(
            name="datatype",
            labels=self.labels,
            active=0
        )

    def fillShelfId(self, shelfId:ShelfId):
        # self.radio.value is the button number that is pressed -> map it to label
        shelfId.datatype = self.labels[self.widget.active]
    
    def registerCallback(self, callback):
        self.widget.on_change('active', callback)

class PlaceholderDatatypeSelector(ShelfIdSelector):
    def fillShelfId(self, shelfId:ShelfId):
        # self.radio.value is the button number that is pressed -> map it to label
        shelfId.datatype = "data"
    def registerCallback(self, callback):
        pass

class ClueParamsSelector(ShelfIdSelector):
    def __init__(self, clueParamList) -> None:
        self.widget = bokeh.models.RadioButtonGroup(
            name="clue_params",
            labels=clueParamList,
            active=0
        )
    
    def fillShelfId(self, shelfId:ShelfId):
        shelfId.clue_param_name = self.widget.labels[self.widget.active]

    def registerCallback(self, callback):
        self.widget.on_change('active', callback)

class PlaceholderClueParamsSelector(ShelfIdSelector):
    def fillShelfId(self, shelfId:ShelfId):
        shelfId.clue_param_name = "default"

import argparse

def parseArgs():
    parser = argparse.ArgumentParser(description="Plotting code to be run using Bokeh server, use bokeh serve SCRIPT.py --args ARGS")
    parser.add_argument("--hist-folder", dest="hist_folder",
        help="path to folder holding all histograms. Will load recursively all clueparams and datatypes inside this folder")
    parser.add_argument("--single-file", dest='single_file',
        help="Only load a single pickle file (for debugging), given by this full path to the file")
    return parser.parse_args()

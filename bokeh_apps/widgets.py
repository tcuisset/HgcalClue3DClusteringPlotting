import bokeh.models

from HistogramLib.bokeh.selector_widgets import *
from HistogramLib.histogram import HistogramKind
from hists.parameters import beamEnergies
from hists.custom_hists import intervalHoldingFractionOfEnergy_energyFraction_axis

def makeLayerSelector():
    return ExplodeSelector(RangeAxisSelector("layer",
        title="Layer selection",
        start=0,
        end=30,
        step=1,
        value=(1,28)
    ))

def makeBeamEnergySelector():
    return ExplodeSelector(MultiSelectAxisSelector("beamEnergy",
        title="Beam energy",
        options=[(str(value), str(value) + " GeV") for value in beamEnergies],
        value=[str(energy) for energy in beamEnergies],
        height=200
    ))

def makeMainOrAllTrackstersSelector():
    """ For CLUE3D, whether we use all 3D clusters per event or only the cluster with the highest energy """
    return ExplodeSelector(RadioButtonGroupAxisSelector("mainOrAllTracksters",
        name="mainOrAllTracksters",
        labels=["allTracksters", "mainTrackster"],
        active=0
    ))

def makeRechitEnergySelector():
    """ Filter rechits energy. Note that it is a log axis, which cannot be implemented in Bokeh sliders, 
    so the selected value is rounded to the nearest bin"""
    return RangeAxisSelector("rechits_energy",
        title="Rechits energy (NB:is binned, logscale)",
        start=0.001,
        end=20,
        step=0.01,
        value=(0.001, 20.),
        logScale=True
    )

def makePointTypeSelector():
    return ExplodeSelector(MultiSelectAxisSelector("pointType",
        title="Point type",
        options=[("0", "follower"), ("1", "seed"), ("2", "outlier")],
        value=["0", "1", "2"],
        height=80
    ))

def makeCluster3DSizeSelector():
    return SliderMinWithOverflowAxisSelector("clus3D_size",
        title="Min 3D cluster size (ie nb of 2D clusters)",
        start=1,
        end=10,
        value=1
    )

def makeIntervalEnergyFractionSelector():
    """ For longitudinal profiles, which fraction of energy to use for determining shortest interval holding fraction of energy"""
    ax = intervalHoldingFractionOfEnergy_energyFraction_axis
    return ExplodeSelector(RadioButtonGroupAxisSelector(ax.name,
        name=ax.name, title="Energy fraction of 3D cluster",
        labels=[ax.bin(i) for i in range(ax.size)],
        active=0
    ))

class HistIdSelectorImpl(Selector):
    selectorType = SelectorType.HISTOGRAM_ID
    def __init__(self, keyName, onChangeValue='value') -> None:
        self.selection = HistogramIdFixedSelection(key=keyName)
        self._updateSelection()
        self.model.on_change(onChangeValue, self._modelCallback)
        self.callbacks = []

    def _modelCallback(self, attr, old, new):
        self._updateSelection()
        for callback in self.callbacks:
            callback(self, plotsHaveChanged=False)
    
    def selections(self) -> List[Selection]:
        return [self.selection]

    def registerCallback(self, callback):
        self.callbacks.append(callback)


class DatatypeSelector(HistIdSelectorImpl, ExplodableSelector):
    def __init__(self, datatypes:list[str]) -> None:
        self.model = bokeh.models.RadioButtonGroup(
            name="datatype",
            labels=datatypes,
            active=0
        )
        self.allSelections = [HistogramIdFixedSelection(key="datatype", value=datatype) for datatype in self.model.labels]
        super().__init__('datatype', onChangeValue='active')
    
    def _updateSelection(self) -> None:
        # self.radio.value is the button number that is pressed -> map it to label
        self.selection.value = self.model.labels[self.model.active]

def makeDatatypeSelector(datatypes:list[str]):
    return ExplodeSelector(DatatypeSelector(datatypes))


class ClueParamsSelector(HistIdSelectorImpl):
    def __init__(self, clueParamList) -> None:
        self.model = bokeh.models.RadioButtonGroup(
            name="clueParamName",
            labels=clueParamList,
            active=0
        )
        super().__init__('clueParamName', onChangeValue='active')
    
    def _updateSelection(self) -> None:
        self.selection.value = self.model.labels[self.model.active]


class HistogramKindRadioButton(HistIdSelectorImpl):
    selectorType = SelectorType.HISTOGRAM_KIND
    labels_dict = {"Count" : HistogramKind.COUNT,
                    "Weight" : HistogramKind.WEIGHT,
                    "Profile" : HistogramKind.PROFILE}

    def __init__(self) -> None:
        self.model = bokeh.models.RadioButtonGroup(
            labels=list(self.labels_dict.keys()),
            active=0
        )
        self.selection = HistKindFixedSelection()
        self._updateSelection()
        self.model.on_change("active", self._modelCallback)
        self.callbacks = []

    def _updateSelection(self) -> None:
        self.selection.kind = self.labels_dict[self.model.labels[self.model.active]]


class DensityHistogramToggle(HistIdSelectorImpl):
    selectorType = SelectorType.DENSITY_HISTOGRAM
    def __init__(self) -> None:
        self.model = bokeh.models.Toggle(
            label="Density",
            active=False
        )
        self.selection = DensityHistogramFixedSelection(self.model.active)
        self._updateSelection()
        self.model.on_change("active", self._modelCallback)
        self.callbacks = []

    def _updateSelection(self) -> None:
        self.selection.density = self.model.active

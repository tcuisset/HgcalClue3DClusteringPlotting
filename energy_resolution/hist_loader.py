import hist

from HistogramLib.store import HistogramStore
from HistogramLib.histogram import HistogramKind
from hists.store import HistogramId


class HistLoader:
    def __init__(self, store:HistogramStore) -> None:
        self.store = store

    def getClue3DProjected(self, datatype, beamEnergy):
        return (
            self.store.get(HistogramId("Clus3DClusteredEnergy", "cmssw", datatype))
            .getHistogram(HistogramKind.COUNT)
            [{"beamEnergy":hist.loc(beamEnergy), "mainOrAllTracksters":hist.loc("mainTrackster")}]
            .project("clus3D_energy")
        )
    def getClue2DProjected(self, datatype, beamEnergy):
        return (
            self.store.get(HistogramId("EnergyClustered2DPerEvent", "cmssw", datatype))
            .getHistogram(HistogramKind.COUNT)
            [{"beamEnergy":hist.loc(beamEnergy)}]
            .project("clus2D_energy_sum")
        )
    def getRechitsProjected(self, datatype, beamEnergy):
        return (
            self.store.get(HistogramId("RechitsTotalEnergyPerEvent", "cmssw", datatype))
            .getHistogram(HistogramKind.COUNT)
            [{"beamEnergy":hist.loc(beamEnergy)}]
            .project("rechits_energy_sum")
        )
    
    def getProjected(self, datatype, beamEnergy, level):
        """ Level can be rechits, clue, clue3d """
        if level == "rechits":
            return self.getRechitsProjected(datatype, beamEnergy)
        elif level == "clue":
            return self.getClue2DProjected(datatype, beamEnergy)
        elif level == "clue3d":
            return self.getClue3DProjected(datatype, beamEnergy)
        else:
            raise ValueError("Level")


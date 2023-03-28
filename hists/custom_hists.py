from functools import partial
import hist

from .dataframe import DataframeComputations, divideByBeamEnergy
from HistogramLib.histogram import MyHistogram, HistogramVariable

beamEnergies = [20, 30, 50, 80, 100, 120, 150, 200, 250, 300]
trueBeamEnergyMap = {20 : 20, 30 : 30, 50 : 49.99, 80 : 79.93, 100 : 99.83, 120 : 119.65, 150 : 149.14, 200 : 197.32, 250 : 243.61, 300 : 287.18}
trueBeamEnergies = trueBeamEnergyMap.values()

# Axises which have sliders using Bokeh for plotting
# Their names must match those of the Bokeh sliders, so they cannot be changed
beamEnergiesAxis = hist.axis.IntCategory(beamEnergies, name="beamEnergy", label="Beam energy (GeV)")
layerAxis = hist.axis.Integer(start=0, stop=30, name="layer", label="Layer number")
# Partially bound constructor to allow overriding label (do not change name if you want the sliders to work)
layerAxis_custom = partial(hist.axis.Integer, start=0, stop=30, name="layer", label="Layer number")
# Makes a growing axis for ntupleNumber (adds bins when needed) (not implemented yet)
#ntupleNumberAxis = hist.axis.IntCategory([], growth=True, name="ntupleNumber", label="ntuple number")

#Bind commonly used parameters of hist axis construction using functools.partial
xyPosition_axis = partial(hist.axis.Regular, bins=150, start=-10., stop=10.)
zPosition_axis = partial(hist.axis.Regular, bins=150, start=0., stop=60.)

# See PointsCloud.h PointsCloud::PointType for which values correspond to what number
pointType_axis = partial(hist.axis.IntCategory, [0, 1, 2], name="pointType",  label="0 : follower, 1 : seed, 2: outlier")

# An axis for difference in position (for example cluster spatial resolution)
# 200 bins takes about 500MB of space with all the other axises (200 * 200 * 10 (beamEnergy) * 30 (layer) * 2 (mainTrackster) * 8 (double storage) = 0.2 GB)
diffX_axis = hist.axis.Regular(bins=100, start=-8., stop=8., name="clus2D_diff_impact_x", label="x position difference (cm)")
diffY_axis = hist.axis.Regular(bins=100, start=-8., stop=8., name="clus2D_diff_impact_y", label="y position difference (cm)")

# Axis for plotting total clustered energy per event
totalClusteredEnergy_axis = partial(hist.axis.Regular, bins=2000, start=0, stop=350)
fractionOfBeamEnergy_axis = partial(hist.axis.Regular, bins=1000, start=0, stop=1.2, label="Fraction of beam energy")

############# IMPACT
class ImpactXY(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, layerAxis,
            xyPosition_axis(name="impactX", label="Impact x position (cm)"),
            xyPosition_axis(name="impactY", label="Impact y position (cm)"),

            label="Impact position (x-y)",
            binCountLabel="Event count",
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.impactWithBeamEnergy)

############ MISC
class TrueBeamEnergy(MyHistogram):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(beamEnergiesAxis,
            hist.axis.Regular(start=0., stop=320., bins=1000, name="trueBeamEnergy", label="True particle momentum (from simulation) (GeV)"),

            label="True beam energy",
            binCountLabel="Event count"
        )
    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.trueBeamEnergy)

############ RECHITS
rechits_energy_profileVariable = HistogramVariable('rechits_energy', 'Mean reconstructed hit energy in a bin (GeV)')
rechits_energy_weightVariable = HistogramVariable('rechits_energy', 'Sum of all rechits energies in a bin (GeV)')

rechits_energy_axis = partial(hist.axis.Regular, bins=50, start=0.002, stop=20., transform=hist.axis.transform.log,
    name="rechits_energy", label="Rechit energy (GeV)")
#For rho use a transform to have more bins at low rho. For now use sqrt, but log would be better (though needs to find a way to include start=0)
rechits_rho_axis = partial(hist.axis.Regular, bins=100, start=0, stop=20., transform=hist.axis.transform.sqrt,
    name="rechits_rho", label="RecHit rho (local energy density) (GeV)")
rechits_delta_axis = partial(hist.axis.Regular, bins=100, start=0, stop=3., name="rechits_delta", label="RecHit delta (distance to nearest higher) (cm)")

# Here rechits_energy is meant as a plot axis so we change its name to rechits_energy_plotAxis so it is not projected on automatically
class RechitsEnergy(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, layerAxis, pointType_axis(),
            hist.axis.Regular(bins=500, start=0.002, stop=20., transform=hist.axis.transform.log,
                name="rechits_energy_plotAxis", label="Rechits energy (GeV)"),
            label="RecHits energy",
            binCountLabel="RecHits count",
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.rechits, {'layer' : "rechits_layer", 'rechits_energy_plotAxis' : 'rechits_energy',
            'pointType':'rechits_pointType'})

# Takes a lot of memory
class RechitsPositionXY(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, layerAxis, pointType_axis(),
            xyPosition_axis(name="rechits_x", label="RecHit x position (cm)", bins=100),
            xyPosition_axis(name="rechits_y", label="RecHit y position (cm)", bins=100),

            label="RecHits position (x-y) (no filter on rechit energy)",
            binCountLabel="RecHits count",
            profileOn=rechits_energy_profileVariable,
            weightOn=rechits_energy_weightVariable
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.rechits, {'layer' : "rechits_layer", 'pointType':'rechits_pointType'})

class RechitsTotalEnergyPerEvent(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, 
            totalClusteredEnergy_axis(name="rechits_energy_sum", label="Total reconstructed energy per event (GeV)"),

            label="Total rechit energy per event",
            binCountLabel="Event count",
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.rechits_totalReconstructedEnergyPerEvent)

class RechitsTotalEnergyFractionPerEvent(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis,
            fractionOfBeamEnergy_axis(name="rechits_energy_sum_fractionOfBeamEnergy", label="Total reconstructed energy per event, as fraction of beam energy"),

            label="Total rechit energy per event (as fraction of beam energy)",
            binCountLabel="Event count",
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.rechits_totalReconstructedEnergyPerEvent)

class RechitsPositionLayer(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, rechits_energy_axis(), pointType_axis(),
            layerAxis_custom(name="rechits_layer", label="RecHit layer number"),

            label="RecHits layer number",
            binCountLabel="RecHits count",
            profileOn=rechits_energy_profileVariable,
            weightOn=rechits_energy_weightVariable
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.rechits, {'pointType':'rechits_pointType'})

class RechitsRho(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, layerAxis, rechits_energy_axis(),  pointType_axis(),
            rechits_rho_axis(),

            label="RecHits rho (local energy density)",
            binCountLabel="RecHits count",
            profileOn=rechits_energy_profileVariable,
            weightOn=rechits_energy_weightVariable
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.rechits, {'layer' : "rechits_layer", 'pointType':'rechits_pointType'})

class RechitsDelta(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, layerAxis, rechits_energy_axis(), pointType_axis(),
            rechits_delta_axis(),

            label="RecHit delta (distance to nearest higher)",
            binCountLabel="RecHits count",
            profileOn=rechits_energy_profileVariable,
            weightOn=rechits_energy_weightVariable
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.rechits, {'layer' : "rechits_layer", 'pointType':'rechits_pointType'})

class RechitsRhoDelta(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, layerAxis,
            rechits_rho_axis(),
            rechits_delta_axis(),

            label="RecHit rho-delta (no filter on rechit energy)",
            binCountLabel="RecHits count",
            profileOn=rechits_energy_profileVariable,
            weightOn=rechits_energy_weightVariable
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.rechits, {'layer' : "rechits_layer"})


class RechitsPointType(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, layerAxis, 
            pointType_axis(name="rechits_pointType"), # we want pointType as plot axis : do not call it pointType
            rechits_energy_axis(),

            label="RecHit Follower/Seed/Outlier",
            binCountLabel="RecHits count",
            profileOn=rechits_energy_profileVariable,
            weightOn=rechits_energy_weightVariable
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.rechits, {'layer' : "rechits_layer"})    


############# 2D clusters ######################
clus2D_energy_axis = partial(hist.axis.Regular, bins=50, start=0.01, stop=70., transform=hist.axis.transform.log,
    name="clus2D_energy", label="2D cluster energy (GeV)")

clus2D_size_axis = partial(hist.axis.Integer, start=1, stop=50, name="clus2D_size", label="2D cluster size ie number of rechits in the cluster")

cluster2D_rho_axis = partial(hist.axis.Regular, bins=100, start=0, stop=100., transform=hist.axis.transform.sqrt,
    name="clus2D_rho", label="2D cluster rho (local energy density) (GeV)")
cluster2D_delta_axis = partial(hist.axis.Regular, bins=100, start=0, stop=3., name="clus2D_delta", label="2D cluster delta (distance to nearest higher) (cm)")

class Clus2DPositionXY(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, layerAxis, pointType_axis(),
            hist.axis.Regular(bins=100, start=-10., stop=10., name="clus2D_x", label="2D cluster x position (cm)"), 
            hist.axis.Regular(bins=100, start=-10., stop=10., name="clus2D_y", label="2D cluster y position (cm)"),
            label = "2D cluster X-Y position",
            binCountLabel="2D clusters count",
            profileOn=HistogramVariable('clus2D_energy', 'Mean of 2D cluster energies in each bin (GeV)'),
            weightOn=HistogramVariable('clus2D_energy', 'Sum of 2D cluster energies in each bin (GeV)')
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters2D, {'layer' : "clus2D_layer", 'pointType' : 'clus2D_pointType'})

class Clus2DPositionLayer(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis,  pointType_axis(),
            layerAxis_custom(name="clus2D_layer", label="2D cluster layer"),
            label = "2D cluster layer",
            binCountLabel="2D clusters count",
            profileOn=HistogramVariable('clus2D_energy', 'Mean of 2D cluster total reconstructed energy in each layer (GeV)'),
            weightOn=HistogramVariable('clus2D_energy', 'Sum of 2D cluster energies in each layer (GeV)')
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters2D, {'pointType' : 'clus2D_pointType'})

class Clus2DSize(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis,  pointType_axis(), layerAxis,
            clus2D_size_axis(),
            label = "2D cluster size",
            binCountLabel="2D clusters count",
            profileOn=HistogramVariable('clus2D_energy', 'Mean of 2D cluster total reconstructed energy in each bin (GeV)'),
            weightOn=HistogramVariable('clus2D_energy', 'Sum of 2D cluster energies in each bin (GeV)')
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters2D, {"layer" : "clus2D_layer", 'pointType' : 'clus2D_pointType'})

# Note : here layer is meant as a plot axis (not to be used with a slider), thus we change its name
class EnergyClustered2DPerLayer(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, layerAxis_custom(name="clus2D_layer"),
            label="Sum of 2D clusters energies per layer",
            binCountLabel="Event count",
            profileOn=HistogramVariable('clus2D_energy_sum_fractionOfBeamEnergy', 'Mean, over all events, of the total 2D clustered energy in each layer, as a fraction of the beam energy'),
            weightOn=HistogramVariable('clus2D_energy_sum', 'Sum (over all events) of all 2D clusters energies in each layer (GeV)')
        )

    def loadFromComp(self, comp:DataframeComputations):
        # no need for mapping as the layerAxis has the right name
        self.fillFromDf(comp.get_clusters2D_perLayerInfo(withBeamEnergy=True).reset_index(level="clus2D_layer")) 

class EnergyClustered2DPerEvent(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis,
            totalClusteredEnergy_axis(name="clus2D_energy_sum", label="Total clustered energy by CLUE2D (GeV)"),
            label="Sum of all 2D clustered energy per event",
            binCountLabel="Event count",
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters2D_totalEnergyPerEvent) 

class FractionEnergyClustered2DPerEvent(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis,
            fractionOfBeamEnergy_axis(name="clus2D_energy_sum_fractionOfBeamEnergy", label="Total clustered energy by CLUE2D, as fraction of beam energy"),
            label="Sum of all 2D clustered energy per event (as fraction of beam energy)",
            binCountLabel="Event count",
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters2D_totalEnergyPerEvent) 

# Note : here layer is meant as a plot axis (not to be used with a slider)
class LayerWithMaximumClustered2DEnergy(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, layerAxis_custom(name="clus2D_layer"), 
            label="Layer with the maximum 2D-clustered energy",
            binCountLabel="Event count",
            profileOn=HistogramVariable('clus2D_energy_sum', 'For each layer, mean of the total 2D clustered energy on this layer for events where this layer is the one with max 2D clustered energy (GeV)'),
            weightOn=HistogramVariable('clus2D_energy_sum', 'For each layer, sum of all 2D clustered energy for events where this layer is the one with max 2D clustered energy (GeV)')
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters2D_sumClustersOnLayerWithMaxClusteredEnergy)

# Note : here layer is meant as a plot axis (not to be used with a slider)
class NumberOf2DClustersPerLayer(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, layerAxis_custom(name="clus2D_layer"), 
            label="Number of 2D clusters per layer",
            binCountLabel="2D clusters count",
            profileOn=HistogramVariable('clus2D_energy', 'Mean of 2D cluster reconstructed energy per layer (for all events and 2D clusters in the layer) (GeV)'),
            weightOn=HistogramVariable('clus2D_energy', 'Sum of the energies of all 2D clusters in the layer (GeV)')
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters2D)

class Cluster2DRho(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, layerAxis, pointType_axis(),
            cluster2D_rho_axis(),
            label="2D cluster rho (local energy density)",
            binCountLabel="2D clusters count",
            profileOn=HistogramVariable('clus2D_energy', 'Mean of 2D cluster reconstructed energy over all 2D clusters which have rho in this bin (GeV)'),
            weightOn=HistogramVariable('clus2D_energy', 'Sum of the energies of all 2D clusters in each rho bin (GeV)')
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters2D, {'layer' : "clus2D_layer", 'pointType' : 'clus2D_pointType'})

class Cluster2DDelta(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, layerAxis, pointType_axis(),
            cluster2D_delta_axis(),
            label="2D cluster delta (distance to nearest higher)",
            binCountLabel="2D clusters count",
            profileOn=HistogramVariable('clus2D_energy', 'Mean of 2D cluster reconstructed energy over all 2D clusters which have delta in this bin (GeV)'),
            weightOn=HistogramVariable('clus2D_energy', 'Sum of the energies of all 2D clusters in each delta bin (GeV)')
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters2D, {'layer' : "clus2D_layer", 'pointType' : 'clus2D_pointType'})

class Cluster2DRhoDelta(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, layerAxis,
            cluster2D_rho_axis(),
            cluster2D_delta_axis(),
            label="2D cluster rho-delta",
            binCountLabel="2D clusters count",
            profileOn=HistogramVariable('clus2D_energy', 'Mean of 2D cluster reconstructed energy over all 2D clusters which have rho&delta in this bin (GeV)'),
            weightOn=HistogramVariable('clus2D_energy', 'Sum of the energies of all 2D clusters in each rho&delta bin (GeV)')
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters2D, {'layer' : "clus2D_layer"})

class Cluster2DPointType(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, layerAxis,
            pointType_axis(name="clus2D_pointType"), # we want pointType as plot axis : do not call it pointType
            clus2D_energy_axis(),

            label="Cluster 2D Follower/Seed/Outlier",
            binCountLabel="Cluster 2D count",
            profileOn=HistogramVariable('clus2D_energy', 'Mean of 2D cluster reconstructed energy over all 2D clusters in this bin (GeV)'),
            weightOn=HistogramVariable('clus2D_energy', 'Sum of the energies of all 2D clusters in each bin (GeV)')
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters2D, {'layer' : "clus2D_layer"})    

###################   3D clusters  ##################
clus3D_mainOrAllTracksters_axis = hist.axis.StrCategory(["allTracksters", "mainTrackster"], name="mainOrAllTracksters",
    label="For 3D clusters, whether to consider for each event all 3D clusters (allTracksters) or only the highest energy 3D cluster (mainTrackster)")

#clus3D_minNumLayerCluster_axis = hist.axis.Integer(start=0, stop=10, name="clus3D_minNumLayerCluster", 
#    label="Minimum number of 2D clusters to make a 3D cluster (not inclusive, ie 5 keeps clusters with 5 2D clusters but not 4 layers)")
cluster3D_size_axis = partial(hist.axis.Integer, start=1, stop=10, name="clus3D_size", label="3D cluster size ie number of 2D clusters that make out this 3D cluster")

class Clus3DPositionXY(MyHistogram):
    def __init__(self) -> None:
        super().__init__(
            beamEnergiesAxis, clus3D_mainOrAllTracksters_axis, cluster3D_size_axis(),
            hist.axis.Regular(bins=100, start=-10., stop=10., name="clus3D_x", label="3D cluster x position (cm)"), 
            hist.axis.Regular(bins=100, start=-10., stop=10., name="clus3D_y", label="3D cluster y position (cm)"),
            label = "3D cluster X-Y position",
            binCountLabel="3D clusters count",
            profileOn=HistogramVariable('clus3D_energy', 'Mean of 3D cluster energies in each bin (GeV)'),
            weightOn=HistogramVariable('clus3D_energy', 'Sum of 3D cluster energies in each bin (GeV)')
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters3D, valuesNotInDf={"mainOrAllTracksters": "allTracksters"})
        self.fillFromDf(comp.clusters3D_largestCluster, valuesNotInDf={"mainOrAllTracksters": "mainTrackster"})

class Clus3DSpatialResolutionUsingLayerWithMax2DEnergyXY(MyHistogram):
    def __init__(self) -> None:
        super().__init__(
            beamEnergiesAxis, clus3D_mainOrAllTracksters_axis, cluster3D_size_axis(),
            hist.axis.Regular(bins=100, start=-10., stop=10., name="clus3D_x", label="3D cluster x-impactX (cm)"), 
            hist.axis.Regular(bins=100, start=-10., stop=10., name="clus3D_y", label="3D cluster y-impactY (cm)"),
            label = "3D cluster X-Y position minus impact position\n(impact at layer with highest 2D clustered energy of each 3D cluster)",
            binCountLabel="3D clusters count",
            profileOn=HistogramVariable('clus3D_energy', 'Mean of 3D cluster energies in each bin (GeV)'),
            weightOn=HistogramVariable('clus3D_energy', 'Sum of 3D cluster energies in each bin (GeV)')
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters3D_impact_usingLayerWithMax2DClusteredEnergy, valuesNotInDf={"mainOrAllTracksters": "allTracksters"})
        self.fillFromDf((comp.clusters3D_impact_usingLayerWithMax2DClusteredEnergy
                .set_index("clus3D_id", append=True)
                .loc[comp.clusters3D_largestClusterIndex]),
            valuesNotInDf={"mainOrAllTracksters": "mainTrackster"})

class Clus3DPositionZ(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, clus3D_mainOrAllTracksters_axis, cluster3D_size_axis(),
            zPosition_axis(name="clus3D_z", label="3D cluster z position (cm)"),
            label = "3D cluster Z position",
            binCountLabel="3D clusters count",
            profileOn=HistogramVariable('clus3D_energy', 'Mean of 3D cluster total reconstructed energy in each z bin (GeV)'),
            weightOn=HistogramVariable('clus3D_energy', 'Sum of 3D cluster energies in each z bin (GeV)')
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters3D, valuesNotInDf={"mainOrAllTracksters": "allTracksters"})
        self.fillFromDf(comp.clusters3D_largestCluster, valuesNotInDf={"mainOrAllTracksters": "mainTrackster"})

# Here clus3D_size is a plot axis
class Clus3DSize(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, clus3D_mainOrAllTracksters_axis,
            cluster3D_size_axis(name="clus3D_size_custom", stop=50),
            label = "3D cluster size",
            binCountLabel="3D clusters count",
            profileOn=HistogramVariable('clus3D_energy', 'Mean of 3D cluster energy in each bin (GeV)'),
            weightOn=HistogramVariable('clus3D_energy', 'Sum of 3D cluster energies in each bin (GeV)')
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters3D, {"clus3D_size_custom" : "clus3D_size"}, valuesNotInDf={"mainOrAllTracksters": "allTracksters"})
        self.fillFromDf(comp.clusters3D_largestCluster, {"clus3D_size_custom" : "clus3D_size"}, valuesNotInDf={"mainOrAllTracksters": "mainTrackster"})


# Warning : Takes a lot of memory (~5GB on file)
class Clus3DSpatialResolutionOf2DClusters(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, layerAxis, cluster3D_size_axis(),
            diffX_axis, diffY_axis,
            label = "Spatial resolution of 2D clusters that are part of a 3D cluster (NB: mainOrAllTrackster not included)",
            binCountLabel="3D clusters count",
            profileOn=HistogramVariable('clus2D_energy', 'Mean of 2D cluster energies in each bin (only 2D clusters that are in a 3D cluster) (GeV)'),
            weightOn=HistogramVariable('clus2D_energy', 'Sum of 2D cluster energies in each bin (only 2D clusters that are in a 3D cluster) (GeV)'),
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters3D_merged_2D_impact,
            mapping={'layer' : "clus2D_layer"}, valuesNotInDf={"mainOrAllTracksters": "allTracksters"})
        self.fillFromDf(comp.clusters3D_merged_2D_impact.loc[comp.clusters3D_largestClusterIndex], 
            mapping={'layer' : "clus2D_layer"}, valuesNotInDf={"mainOrAllTracksters": "mainTrackster"})

# Note : layer is meant as an axis (not a slider to project on)
class Clus3DFirstLayerOfCluster(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, clus3D_mainOrAllTracksters_axis, cluster3D_size_axis(),
            layerAxis_custom(name="clus2D_minLayer", label="First layer clustered in CLUE3D"),
            label = "First layer used by CLUE3D",
            binCountLabel="3D clusters count",
            profileOn=HistogramVariable('clus3D_energy', 'Mean of the 3D cluster total reconstructed energy whose first layer is in the layer bin (GeV)'),
            weightOn=HistogramVariable('clus3D_energy', 'Sum of the 3D cluster total reconstructed energy for all clusters whose first layer is in the layer bin (GeV)'),
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters3D_firstLastLayer, 
            valuesNotInDf={"mainOrAllTracksters": "allTracksters"})
        self.fillFromDf(comp.clusters3D_firstLastLayer.loc[comp.clusters3D_largestClusterIndex],
            valuesNotInDf={"mainOrAllTracksters": "mainTrackster"})

# Note : layer is meant as an axis (not a slider to project on)
class Clus3DLastLayerOfCluster(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, clus3D_mainOrAllTracksters_axis, cluster3D_size_axis(),
            layerAxis_custom(name="clus2D_maxLayer", label="Last layer clustered in CLUE3D"),
            label = "Last layer used by CLUE3D",
            binCountLabel="3D clusters count",
            profileOn=HistogramVariable('clus3D_energy', 'Mean of the 3D cluster total reconstructed energy whose last layer is in the layer bin (GeV)'),
            weightOn=HistogramVariable('clus3D_energy', 'Sum of the 3D cluster total reconstructed energy for all clusters whose last layer is in the layer bin (GeV)'),
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters3D_firstLastLayer,
            valuesNotInDf={"mainOrAllTracksters": "allTracksters"})
        self.fillFromDf(comp.clusters3D_firstLastLayer.loc[comp.clusters3D_largestClusterIndex],
            valuesNotInDf={"mainOrAllTracksters": "mainTrackster"})



# Note : here layer is meant as a plot axis (not to be used with a slider)
class Clus3DNumberOf2DClustersPerLayer(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, clus3D_mainOrAllTracksters_axis, cluster3D_size_axis(),
            layerAxis_custom(name="clus2D_layer"), 
            label="Number of 2D clusters per layer clustered by CLUE3D",
            profileOn=HistogramVariable('clus2D_energy', 'Mean of clus2D energy of all the 2D clusters member of a 3D cluster that are in this layer (GeV)'),
            weightOn=HistogramVariable('clus2D_energy', 'Sum of clus2D energy of all the 2D clusters member of a 3D cluster that are in this layer (GeV)'),
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters3D_merged_2D, 
            valuesNotInDf={"mainOrAllTracksters": "allTracksters"})
        self.fillFromDf(comp.clusters3D_merged_2D.loc[comp.clusters3D_largestClusterIndex], 
            valuesNotInDf={"mainOrAllTracksters": "mainTrackster"})

# Note : here layer is meant as a plot axis (not to be used with a slider)
class Clus3DClusteredEnergyPerLayer(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, clus3D_mainOrAllTracksters_axis, cluster3D_size_axis(),
            layerAxis_custom(name="clus2D_layer"), 
            label="Clustered energy by CLUE3D per layer",
            profileOn=HistogramVariable('clus2D_energy_sum', 'Mean of the 2D clustered energy by CLUE3D for each layer and each 3D cluster (GeV)'),
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters3D_energyClusteredPerLayer.reset_index(level="clus2D_layer"), 
            valuesNotInDf={"mainOrAllTracksters": "allTracksters"})
        self.fillFromDf(comp.clusters3D_energyClusteredPerLayer.reset_index(level="clus2D_layer").loc[comp.clusters3D_largestClusterIndex], 
            valuesNotInDf={"mainOrAllTracksters": "mainTrackster"})

class Clus3DClusteredEnergy(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, clus3D_mainOrAllTracksters_axis, cluster3D_size_axis(),
            totalClusteredEnergy_axis(name="clus3D_energy", label="3D cluster energy (GeV)"),
            label="Clustered energy by CLUE3D",
            binCountLabel="3D cluster count"
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters3D, 
            valuesNotInDf={"mainOrAllTracksters": "allTracksters"})
        self.fillFromDf(comp.clusters3D_largestCluster, 
            valuesNotInDf={"mainOrAllTracksters": "mainTrackster"})

class Clus3DClusteredFractionEnergy(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, clus3D_mainOrAllTracksters_axis, cluster3D_size_axis(),
            fractionOfBeamEnergy_axis(name="clus3D_energy_fractionOfBeamEnergy", label="3D cluster energy (fraction of beam energy)"),
            label="Clustered energy by CLUE3D per event",
            binCountLabel="3D cluster count"
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(divideByBeamEnergy(comp.clusters3D, "clus3D_energy"), 
            valuesNotInDf={"mainOrAllTracksters": "allTracksters"})
        self.fillFromDf(divideByBeamEnergy(comp.clusters3D_largestCluster, "clus3D_energy"), 
            valuesNotInDf={"mainOrAllTracksters": "mainTrackster"})

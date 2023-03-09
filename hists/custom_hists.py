from functools import partial
import hist

from .dataframe import DataframeComputations
from HistogramLib.histogram import MyHistogram, HistogramVariable

beamEnergies = [20, 50, 80, 100, 120, 150, 200, 250, 300]

# Axises which have sliders using Bokeh for plotting
# Their names must match those of the Bokeh sliders, so they cannot be changed
beamEnergiesAxis = hist.axis.IntCategory(beamEnergies, name="beamEnergy", label="Beam energy (GeV)")
layerAxis = hist.axis.Integer(start=0, stop=30, name="layer", label="Layer number")
# Partially bound constructor to allow overriding label (do not change name if you want the sliders to work)
layerAxis_custom = partial(hist.axis.Integer, start=0, stop=30, name="layer", label="Layer number")
# Makes a growing axis for ntupleNumber (adds bins when needed) (not implemented yet)
#ntupleNumberAxis = hist.axis.IntCategory([], growth=True, name="ntupleNumber", label="ntuple number")

#Bind commonly used parameters of hist axis construction using functools.partial
xyPosition_axis = partial(hist.axis.Regular, bins=200, start=-10., stop=10.)
zPosition_axis = partial(hist.axis.Regular, bins=200, start=0., stop=60.)

#For rho use a transform to have more bins at low rho. For now use sqrt, but log would be better (though needs to find a way to include start=0)
rechits_rho_axis = partial(hist.axis.Regular, bins=100, start=0, stop=20., transform=hist.axis.transform.sqrt,
    name="rechits_rho", label="RecHit rho (local energy density)")
rechits_delta_axis = partial(hist.axis.Regular, bins=100, start=0, stop=3., name="rechits_delta", label="RecHit delta (distance to nearest higher)")
seed_axis = partial(hist.axis.IntCategory, [0, 1]) #0: not a seed, 1: is a seed

cluster2D_rho_axis = partial(hist.axis.Regular, bins=100, start=0, stop=100., transform=hist.axis.transform.sqrt,
    name="clus2D_rho", label="2D cluster rho (local energy density)")
cluster2D_delta_axis = partial(hist.axis.Regular, bins=100, start=0, stop=3., name="clus2D_delta", label="2D cluster delta (distance to nearest higher)")

cluster3D_size_axis = hist.axis.Integer(1, 40, name="cluster3dSize", label="3D cluster size")


# An axis for difference in position (for example cluster spatial resolution)
# 200 bins takes about 500MB of space with all the other axises (200 * 200 * 10 (beamEnergy) * 30 (layer) * 2 (mainTrackster) * 8 (double storage) = 0.2 GB)
diffX_axis = hist.axis.Regular(bins=100, start=-8., stop=8., name="clus2D_diff_impact_x", label="x position difference")
diffY_axis = hist.axis.Regular(bins=100, start=-8., stop=8., name="clus2D_diff_impact_y", label="y position difference")


clus3D_mainOrAllTracksters_axis = hist.axis.StrCategory(["allTracksters", "mainTrackster"], name="mainOrAllTracksters",
    label="For 3D clusters, whether to consider for each event all 3D clusters (allTracksters) or only the highest energy 3D cluster (mainTrackster)")



############# IMPACT
class ImpactXY(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, 
            xyPosition_axis(name="impactX", label="RecHit x position"),
            xyPosition_axis(name="impactY", label="RecHit y position"),

            label="Impact position (x-y)",
            binCountLabel="Event count",
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.impactWithBeamEnergy)

############ RECHITS
rechits_energy_profileVariable = HistogramVariable('rechits_energy', 'Mean reconstructed hit energy in a bin')
rechits_energy_weightVariable = HistogramVariable('rechits_energy', 'Sum of all rechits energies in a bin')

class RechitsPositionXY(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, layerAxis, 
            xyPosition_axis(name="rechits_x", label="RecHit x position"),
            xyPosition_axis(name="rechits_y", label="RecHit y position"),

            label="RecHits position (x-y)",
            binCountLabel="RecHits count",
            profileOn=rechits_energy_profileVariable,
            weightOn=rechits_energy_weightVariable
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.rechits, {'layer' : "rechits_layer"})

class RechitsPositionZ(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis,
            zPosition_axis(name="rechits_z", label="RecHit z position"),

            label="RecHits z position",
            binCountLabel="RecHits count",
            profileOn=rechits_energy_profileVariable,
            weightOn=rechits_energy_weightVariable
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.rechits, {'layer' : "rechits_layer"})

class RechitsRho(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, layerAxis,
            rechits_rho_axis(),

            label="RecHits rho (local energy density)",
            binCountLabel="RecHits count",
            profileOn=rechits_energy_profileVariable,
            weightOn=rechits_energy_weightVariable
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.rechits, {'layer' : "rechits_layer"})

class RechitsDelta(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, layerAxis,
            rechits_delta_axis(),

            label="RecHit delta (distance to nearest higher)",
            binCountLabel="RecHits count",
            profileOn=rechits_energy_profileVariable,
            weightOn=rechits_energy_weightVariable
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.rechits, {'layer' : "rechits_layer"})

class RechitsRhoDelta(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, layerAxis,
            rechits_rho_axis(),
            rechits_delta_axis(),

            label="RecHit rho-delta",
            binCountLabel="RecHits count",
            profileOn=rechits_energy_profileVariable,
            weightOn=rechits_energy_weightVariable
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.rechits, {'layer' : "rechits_layer"})

############# 2D clusters ######################
# Note : here layer is meant as a plot axis (not to be used with a slider), thus we change its name
class EnergyClustered2DPerLayer(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, layerAxis_custom(name="clus2D_layer"), 
            label="Sum of 2D clusters energies per layer",
            binCountLabel="Event count",
            profileOn=HistogramVariable('clus2D_energy_sum', 'Mean, over all events, of the sum of 2D clusters energies in each layer'),
            weightOn=HistogramVariable('clus2D_energy_sum', 'Sum of all 2D clusters energies in each layer')
        )

    def loadFromComp(self, comp:DataframeComputations):
        # no need for mapping as the layerAxis has the right name
        self.fillFromDf(comp.get_clusters2D_perLayerInfo(withBeamEnergy=True).reset_index(level="clus2D_layer")) 

# Note : here layer is meant as a plot axis (not to be used with a slider)
class LayerWithMaximumClustered2DEnergy(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, layerAxis_custom(name="clus2D_layer"), 
            label="Layer with the maximum 2D-clustered energy",
            binCountLabel="Event count",
            profileOn=HistogramVariable('clus2D_energy_sum', 'For each layer, mean of the total 2D clustered energy on this layer for events where this layer is the one with max 2D clustered energy'),
            weightOn=HistogramVariable('clus2D_energy_sum', 'For each layer, sum of all 2D clustered energy for events where this layer is the one with max 2D clustered energy')
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters2D_sumClustersOnLayerWithMaxClusteredEnergy)

# Note : here layer is meant as a plot axis (not to be used with a slider)
class NumberOf2DClustersPerLayer(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, layerAxis_custom(name="clus2D_layer"), 
            label="Number of 2D clusters per layer",
            binCountLabel="2D clusters count",
            profileOn=HistogramVariable('clus2D_energy', 'Mean of 2D cluster reconstructed energy per layer (for all events and 2D clusters in the layer)'),
            weightOn=HistogramVariable('clus2D_energy', 'Sum of the energies of all 2D clusters in the layer')
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters2D)

class Cluster2DRho(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, layerAxis,
            cluster2D_rho_axis(),
            label="2D cluster rho (local energy density)",
            binCountLabel="2D clusters count",
            profileOn=HistogramVariable('clus2D_energy', 'Mean of 2D cluster reconstructed energy over all 2D clusters which have rho in this bin'),
            weightOn=HistogramVariable('clus2D_energy', 'Sum of the energies of all 2D clusters in each rho bin')
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters2D, {'layer' : "clus2D_layer"})

class Cluster2DDelta(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, layerAxis,
            cluster2D_delta_axis(),
            label="2D cluster delta (distance to nearest higher)",
            binCountLabel="2D clusters count",
            profileOn=HistogramVariable('clus2D_energy', 'Mean of 2D cluster reconstructed energy over all 2D clusters which have delta in this bin'),
            weightOn=HistogramVariable('clus2D_energy', 'Sum of the energies of all 2D clusters in each delta bin')
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters2D, {'layer' : "clus2D_layer"})

class Cluster2DRhoDelta(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, layerAxis,
            cluster2D_rho_axis(),
            cluster2D_delta_axis(),
            label="2D cluster rho-delta",
            binCountLabel="2D clusters count",
            profileOn=HistogramVariable('clus2D_energy', 'Mean of 2D cluster reconstructed energy over all 2D clusters which have rho&delta in this bin'),
            weightOn=HistogramVariable('clus2D_energy', 'Sum of the energies of all 2D clusters in each rho&delta bin')
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters2D, {'layer' : "clus2D_layer"})


###################   3D clusters  ##################
class Clus3DPositionXY(MyHistogram):
    def __init__(self) -> None:
        super().__init__(
            beamEnergiesAxis, clus3D_mainOrAllTracksters_axis,
            hist.axis.Regular(bins=100, start=-10., stop=10., name="clus3D_x", label="3D cluster x position"), 
            hist.axis.Regular(bins=100, start=-10., stop=10., name="clus3D_y", label="3D cluster y position"),
            label = "3D cluster X-Y position",
            binCountLabel="3D clusters count",
            profileOn=HistogramVariable('clus3D_energy', 'Mean of 3D cluster energies in each bin'),
            weightOn=HistogramVariable('clus3D_energy', 'Sum of 3D cluster energies in each bin')
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters3D, valuesNotInDf={"mainOrAllTracksters": "allTracksters"})
        self.fillFromDf(comp.clusters3D_largestCluster, valuesNotInDf={"mainOrAllTracksters": "mainTrackster"})

class Clus3DPositionZ(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, clus3D_mainOrAllTracksters_axis,
            zPosition_axis(name="clus3D_z", label="3D cluster z position"),
            label = "3D cluster Z position",
            binCountLabel="3D clusters count",
            profileOn=HistogramVariable('clus3D_energy', 'Mean of 3D cluster total reconstructed energy in each z bin'),
            weightOn=HistogramVariable('clus3D_energy', 'Sum of 3D cluster energies in each z bin')
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters3D, valuesNotInDf={"mainOrAllTracksters": "allTracksters"})
        self.fillFromDf(comp.clusters3D_largestCluster, valuesNotInDf={"mainOrAllTracksters": "mainTrackster"})

class Clus3DSpatialResolution(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, layerAxis, clus3D_mainOrAllTracksters_axis,
            diffX_axis, diffY_axis,
            label = "Spatial resolution of 2D clusters that are part of a 3D cluster",
            binCountLabel="3D clusters count",
            profileOn=HistogramVariable('clus2D_energy', 'Mean of 2D cluster energies in each bin (only 2D clusters that are in a 3D cluster)'),
            weightOn=HistogramVariable('clus2D_energy', 'Sum of 2D cluster energies in each bin (only 2D clusters that are in a 3D cluster)'),
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters3D_merged_2D_impact,
            mapping={'layer' : "clus2D_layer"}, valuesNotInDf={"mainOrAllTracksters": "allTracksters"})
        self.fillFromDf(comp.clusters3D_merged_2D_impact.loc[comp.clusters3D_largestClusterIndex], 
            mapping={'layer' : "clus2D_layer"}, valuesNotInDf={"mainOrAllTracksters": "mainTrackster"})

# Note : layer is meant as an axis (not a slider to project on)
class Clus3DFirstLayerOfCluster(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, clus3D_mainOrAllTracksters_axis,
            layerAxis_custom(name="clus2D_minLayer", label="First layer clustered in CLUE3D"),
            label = "First layer used by CLUE3D",
            binCountLabel="3D clusters count",
            profileOn=HistogramVariable('clus3D_energy', 'Mean of the 3D cluster total reconstructed energy whose first layer is in the layer bin'),
            weightOn=HistogramVariable('clus3D_energy', 'Sum of the 3D cluster total reconstructed energy for all clusters whose first layer is in the layer bin'),
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters3D_firstLastLayer, 
            valuesNotInDf={"mainOrAllTracksters": "allTracksters"})
        self.fillFromDf(comp.clusters3D_firstLastLayer.loc[comp.clusters3D_largestClusterIndex],
            valuesNotInDf={"mainOrAllTracksters": "mainTrackster"})

# Note : layer is meant as an axis (not a slider to project on)
class Clus3DLastLayerOfCluster(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, clus3D_mainOrAllTracksters_axis,
            layerAxis_custom(name="clus2D_maxLayer", label="Last layer clustered in CLUE3D"),
            label = "Last layer used by CLUE3D",
            binCountLabel="3D clusters count",
            profileOn=HistogramVariable('clus3D_energy', 'Mean of the 3D cluster total reconstructed energy whose last layer is in the layer bin'),
            weightOn=HistogramVariable('clus3D_energy', 'Sum of the 3D cluster total reconstructed energy for all clusters whose last layer is in the layer bin'),
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters3D_firstLastLayer,
            valuesNotInDf={"mainOrAllTracksters": "allTracksters"})
        self.fillFromDf(comp.clusters3D_firstLastLayer.loc[comp.clusters3D_largestClusterIndex],
            valuesNotInDf={"mainOrAllTracksters": "mainTrackster"})



# Note : here layer is meant as a plot axis (not to be used with a slider)
class Clus3DNumberOf2DClustersPerLayer(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, clus3D_mainOrAllTracksters_axis,
            layerAxis_custom(name="clus2D_layer"), 
            label="Number of 2D clusters per layer clustered by CLUE3D",
            profileOn=HistogramVariable('clus2D_energy', 'Mean of clus2D energy of all the 2D clusters member of a 3D cluster that are in this layer'),
            weightOn=HistogramVariable('clus2D_energy', 'Sum of clus2D energy of all the 2D clusters member of a 3D cluster that are in this layer'),
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters3D_merged_2D, 
            valuesNotInDf={"mainOrAllTracksters": "allTracksters"})
        self.fillFromDf(comp.clusters3D_merged_2D.loc[comp.clusters3D_largestClusterIndex], 
            valuesNotInDf={"mainOrAllTracksters": "mainTrackster"})

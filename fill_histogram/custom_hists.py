from functools import partial
from HistogramLib.histogram import *

beamEnergies = [20, 50, 80, 100, 120, 150, 200, 250, 300]
beamEnergiesAxis = hist.axis.IntCategory(beamEnergies, name="beamEnergy", label="Beam energy (GeV)")
layerAxis = hist.axis.Integer(start=0, stop=30, name="layer", label="Layer number")
# Makes a growing axis for ntupleNumber (adds bins when needed)
ntupleNumberAxis = hist.axis.IntCategory([], growth=True, name="ntupleNumber", label="ntuple number")

#Bind commonly used parameters of hist axis construction using functools.partial
xyPosition_axis = partial(hist.axis.Regular, bins=50, start=-10., stop=10.)
zPosition_axis = partial(hist.axis.Regular, bins=50, start=0., stop=60.)

rho_axis = hist.axis.Regular(bins=100, start=0, stop=10., name="rho", label="Energy density")
delta_axis = hist.axis.Regular(bins=100, start=0, stop=3., name="delta", label="Distance to nearest higher")
seed_axis = hist.axis.IntCategory([0, 1], name="isSeed") #0: not a seed, 1: is a seed

cluster2dEnergy_axis = hist.axis.Regular(100, 0, 1, name="cluster2dEnergy", label="2D cluster energy")

cluster3dSizeAxis = hist.axis.Integer(1, 40, name="cluster3dSize", label="3D cluster size")
cluster3dEnergyAxis = hist.axis.Regular(100, 0, 20, name="cluster3dEnergy", label="3D cluster energy")


# An axis for difference in position (for example cluster spatial resolution)
diffX_axis = hist.axis.Regular(bins=50, start=-5., stop=5., name="clus2D_diff_impact_x", label="x position difference")
diffY_axis = hist.axis.Regular(bins=50, start=-5., stop=5., name="clus2D_diff_impact_y", label="y position difference")

energy_axis = hist.axis.Regular(bins=100, start=0., stop=10., name="energy", label="Reconstructed hit energy")

totalRecHitEnergy_axis = hist.axis.Regular(bins=500, start=0, stop=300, name="totalRecHitEnergy", label="Total RecHit energy per event (GeV)")



class RechitsPositionXY(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, layerAxis, 
            xyPosition_axis(name="rechits_x", label="RecHit x position"),
            xyPosition_axis(name="rechits_y", label="RecHit y position"),

            label="RecHits position",
            profileOn = ProfileVariable('rechits_energy', 'Reconstructed hit energy')
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.rechits, {'layer' : "rechits_layer"})

class Clus3DPositionXY(MyHistogram):
    def __init__(self) -> None:
        super().__init__(
            beamEnergiesAxis, 
            hist.axis.Regular(bins=50, start=-10., stop=10., name="clus3D_x", label="3D cluster x position"), 
            hist.axis.Regular(bins=50, start=-10., stop=10., name="clus3D_y", label="3D cluster y position"),

            label = "3D cluster X-Y position",
            profileOn = ProfileVariable('clus3D_energy', '3D cluster total reconstructed energy')
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters3D)

class Clus3DPositionZ(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, 
            zPosition_axis(name="clus3D_z", label="3D cluster z position"),
            
            label = "3D cluster X-Y position",
            profileOn = ProfileVariable('clus3D_energy', '3D cluster total reconstructed energy')
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters3D)

class Clus3DSpatialResolution(MyHistogram):
    def __init__(self) -> None:
        super().__init__(beamEnergiesAxis, layerAxis, diffX_axis, diffY_axis,
            label = "Spatial resolution of 2D clusters that are part of a 3D cluster",
            profileOn = ProfileVariable('clus2D_energy', '2D cluster energy')
        )

    def loadFromComp(self, comp:DataframeComputations):
        self.fillFromDf(comp.clusters3D_merged_2D_impact, {
            'layer' : "clus2D_layer"
        })


# #considering only highest energy 3D cluster per event
# hist_dict["clue3d_clusters_spatial_res_largest_3D_cluster"] = hist.Hist(beamEnergiesAxis, layerAxis, cluster2dEnergy_axis, diffX_axis, diffY_axis)

# #Total clustered energy per layer and per 3D cluster
# hist_dict["clue3d_total_clustered_energy"] = hist.Hist(beamEnergiesAxis, layerAxis, cluster2dEnergy_axis)
# #Same but only consider for each event the 3D cluster with the highest energy
# hist_dict["clue3d_total_clustered_energy_by_largest_cluster"] = hist.Hist(beamEnergiesAxis, layerAxis, cluster2dEnergy_axis)

import sys
sys.path.append("Plotting/")
from vis_clue3D import Clue3DVisualization
from vis_layer import LayerVisualization
from utils import *

el = EventLoader('/data_cms_upgrade/cuisset/testbeam18/clue3d/v31/cmssw/data/CLUE_clusters.root')
event = el.loadEvent(EventID(150, 496, 18369))


fig = (Clue3DVisualization(event)
    .add3DClusters()
    .add2DClusters()
    .addRechits()
    .addImpactTrajectory()
    .fig)
fig.show()

vis = (LayerVisualization(event, layerNb=13) #layer 13 (weird clus pos), layer 10 : double layer cluster
    .add2DClusters()
    .addRechits()
    #.addImpactPoint()
    #.addCircleSearchForComputingClusterPosition()
    )

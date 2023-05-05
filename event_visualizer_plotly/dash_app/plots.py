from event_visualizer_plotly.utils import LoadedEvent
from event_visualizer_plotly.vis_clue3D import Clue3DVisualization
from event_visualizer_plotly.vis_layer import LayerVisualization
from event_visualizer_plotly.vis_longitudinal_profile import LongitudinalProfileVisualization


def makePlotClue3D(event:LoadedEvent):
    """ Returns a Plotly figure representing the CLUE3D vis from a loaded event """
    fig = (Clue3DVisualization(event)
        .addDetectorCylinder()
        .addRechits()
        .add2DClusters()
        .add3DClusters()
        .addImpactTrajectory()
        .addSliders()
    ).fig
    fig.update_layout(dict(uirevision=1)) # Keep the current view in any case. See https://community.plotly.com/t/preserving-ui-state-like-zoom-in-dcc-graph-with-uirevision-with-dash/15793
    return fig

def makePlotLayer(event:LoadedEvent, layer:int):
    """ Returns a Plotly figure representing the per-layer vis from a loaded event """
    fig = (LayerVisualization(event, layerNb=layer)
        .add2DClusters()
        .addRechits()
        .addImpactPoint()
        .addCircleSearchForComputingClusterPosition()
    ).fig
    fig.update_layout(dict(uirevision=1))
    return fig

def makePlotLongitudinalProfile(event:LoadedEvent):
    """ Returns a Plotly figure representing the longitudinal profile from a loaded event """
    fig = (LongitudinalProfileVisualization(event)
        .addRechitsProfile()
        .addClueProfile()
    ).fig
    fig.update_layout(dict(uirevision=1))
    return fig

""" Functions making the plotly plots for the dash app, linking vis_* code to dash """

from event_visualizer.event_index import LoadedEvent
from event_visualizer.plotter.clue3D import Clue3DVisualization
from event_visualizer.plotter.layer import LayerVisualization
from event_visualizer.plotter.longitudinal_profile import LongitudinalProfileVisualization

zAxisDropdownSettings = {"layer-3": "Layer as z", "layer-5":"Layer as z (extra space)", "z-1" : "z - aspect ratio=1", "z-3" : "z - aspect ratio 3", "z-5":"z - aspect ratio 5"}
zAxisDropdownSettingsMap = {"layer-3": dict(zAspectRatio=3, useLayerAsZ=True), "layer-5": dict(zAspectRatio=5, useLayerAsZ=True),
    "z-1" : dict(zAspectRatio=1, useLayerAsZ=False),  "z-3" : dict(zAspectRatio=3, useLayerAsZ=False), "z-5" : dict(zAspectRatio=5, useLayerAsZ=False)}

def makePlotClue3D(event:LoadedEvent, zAxisSetting, projectionType):
    """ Returns a Plotly figure representing the CLUE3D vis from a loaded event """
    try:
        plotSettings = zAxisDropdownSettingsMap[zAxisSetting]
    except KeyError as e:
        print(e)
        plotSettings = dict()
    
    fig = (Clue3DVisualization(event, projection=projectionType, **plotSettings)
        .addDetectorCylinder()
        .addRechits()
        .add2DClusters()
        .add3DClusters()
        .addImpactTrajectory()
        #.addSliders()
    ).fig
    fig.update_layout(dict(uirevision=1)) # Keep the current view in any case. See https://community.plotly.com/t/preserving-ui-state-like-zoom-in-dcc-graph-with-uirevision-with-dash/15793
    return fig

def makePlotLayer(event:LoadedEvent, layer:int):
    """ Returns a Plotly figure representing the per-layer vis from a loaded event """
    fig = (LayerVisualization(event, layerNb=layer)
        .add2DClusters()
        .addRechits()
        .addImpactPoint()
        .addButtons()
        .addDetectorExtent()
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

import argparse
import urllib.parse

import dash
from dash import Dash, html, Input, Output, State, dcc
from dash.exceptions import PreventUpdate
import uproot
import awkward as ak

from hists.parameters import beamEnergies, ntupleNumbersPerBeamEnergy
from event_visualizer_plotly.utils import EventLoader, EventID, LoadedEvent
from event_visualizer_plotly.vis_clue3D import Clue3DVisualization
from event_visualizer_plotly.vis_layer import LayerVisualization
from event_visualizer_plotly.vis_longitudinal_profile import LongitudinalProfileVisualization

parser = argparse.ArgumentParser(
    prog="dash_event_visualizer",
    description="CLUE3D event visualizer using Dash",
)
parser.add_argument('-p', '--port', default=8080, help="Port to listen on")
parser.add_argument('-i', '--input-file', default="/eos/user/t/tcuisset/hgcal/testbeam18-clue3d/v33/cmssw/data/CLUE_clusters.root",
    dest="input_file", help="Path to CLUE_clusters.root file (included)")
parser.add_argument('-d', '--debug', action=argparse.BooleanOptionalAction, help="Enable debug mode", dest="debug", default=True)
parser.add_argument('-H', '--host', dest="host", default=None,
    help="Host name to bin on (if not specified, use Dash default  which is env variable HOST or 127.0.0.1). On llruicms you should put 'llruicms01'")
args = parser.parse_args()

# Local : /data_cms_upgrade/cuisset/testbeam18/clue3d/v33/cmssw/data/CLUE_clusters.root
eventLoader = EventLoader(args.input_file)

app = Dash(__name__)

legendDivStyle = {'flex': '0 1 auto', 'margin':"10px"}
app.layout = html.Div([
    html.Div([
        dcc.Location(id="url", refresh=False),
        html.H1(children='CLUE3D event visualizer (right-click to rotate, left-click to move)'),
        html.Div(children=[
            html.Div("Beam energy (GeV) :", style=legendDivStyle),
            dcc.Dropdown(options=beamEnergies, value=20, id="beamEnergy", style={'flex': '1 1 auto'}),
            html.Div("Ntuple number :", style=legendDivStyle),
            dcc.Dropdown(id="ntupleNumber", style={'flex': '1 1 auto'}),
            html.Div("Event :", style=legendDivStyle),
            dcc.Dropdown(id="event", style={'flex': '1 1 auto'}),
            html.Div("Layer (for layer view) :"),
            dcc.Dropdown(options=list(range(1, 29)), id="layer", value=1),
        ], style={"display":"flex", "flex-flow":"row"}),
    ], style={'flex': '0 1 auto'}),
    dcc.Tabs([
        dcc.Tab(label="3D view", children=[
            dcc.Graph(id="plot_3D", style={"height":"100%"}),
        ]),
        dcc.Tab(label="Layer view", children=[
            dcc.Graph(id="plot_layer", style={"height":"100%"})
        ]),
        dcc.Tab(label="Longitudinal profile", children=[
            dcc.Graph(id="plot_longitudinal-profile", style={"height":"100%"})
        ]),
    ], parent_style={'flex': '1 1 auto'}, content_style={'flex': '1 1 auto'})
    
], style={'display': 'flex', 'flex-flow': 'column', "height":"100vh"})

@app.callback(
    Output("ntupleNumber", "options"),
    [Input("beamEnergy", "value")]
)
def update_ntupleNumber(beamEnergy):
    return ntupleNumbersPerBeamEnergy[beamEnergy]

@app.callback(
    Output("event", "options"),
    [Input("ntupleNumber", "value")]
)
def update_available_events(ntupleNumber):
    return eventLoader.tree.arrays(cut=f"(ntupleNumber == {ntupleNumber})", filter_name=["event"]).event


def makePlotClue3D(event:LoadedEvent):
    """ Returns a Plotly figure representing the CLUE3D vis from a loaded event """
    fig = (Clue3DVisualization(event)
        .add3DClusters()
        .add2DClusters()
        .addRechits()
        .addImpactTrajectory()
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

def figureOutUrlUpdates(ntupleNumber, eventNb, urlSearchValue):
    trig_id = dash.callback_context.triggered_id
    if trig_id == "url" or trig_id is None:
        # Update inputs from URL value
        parsed_url_query = urllib.parse.parse_qs(urlSearchValue[1:]) # Drop the leading "?"
        try:
            return parsed_url_query["ntuple"], parsed_url_query["event"], urlSearchValue
        except KeyError:
            raise PreventUpdate()
    else:
        # Update URL from inputs
        return ntupleNumber, eventNb, "?"+urllib.parse.urlencode({"event":eventNb, "ntuple":ntupleNumber})

def loadEvent(ntuple, event) -> LoadedEvent:
    if ntuple is not None and event is not None:
        return eventLoader.loadEvent(EventID(ntuple, event))
    raise RuntimeError()

@app.callback(
    [Output("ntupleNumber", "value"), Output("event", "value"), Output("url", "search"),
    Output("plot_3D", "figure"), Output("plot_layer", "figure"), Output("plot_longitudinal-profile", "figure")],
    [Input("ntupleNumber", "value"), Input("event", "value"), Input("url", "search"), State("layer", "value")]
)
def mainEventUpdate(ntupleNumber, eventNb, urlSearchValue, layer):
    """ Main callback to update all the plots at the same time.
    layer is State as there is another callback updateOnlyLayerPlot to update just the layer view """
    ntupleNumber, eventNb, urlSearchValue = figureOutUrlUpdates(ntupleNumber, eventNb, urlSearchValue)
    
    try:
        event = loadEvent(ntupleNumber, eventNb)
        plot_3D = makePlotClue3D(event)
        plot_layer = makePlotLayer(event, layer)
        plot_longitudinal = makePlotLongitudinalProfile(event)
    except RuntimeError:
        plot_3D = None
        plot_layer = None
        plot_longitudinal = None
    
    return ntupleNumber, eventNb, urlSearchValue, plot_3D, plot_layer, plot_longitudinal


@app.callback(
    Output("plot_layer", "figure", allow_duplicate=True),
    [State("ntupleNumber", "value"), State("event", "value"), Input("layer", "value")],
    prevent_initial_call=True, # On intial loading the layer view is loaded by the main callback mainEventUpdate
)
def updateOnlyLayerPlot(ntupleNumber, eventNb, layer):
    """ Small callback to only update the layer view """
    try:
        event = loadEvent(ntupleNumber, eventNb)
    except RuntimeError:
        return None
    return makePlotLayer(event, layer)

if __name__ == '__main__':
    if args.host is None:
        app.run(debug=args.debug, port=args.port)
    else:
        app.run(debug=args.debug, port=args.port, host=args.host)
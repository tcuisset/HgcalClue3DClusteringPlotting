""" Event visualizer using Dash
Environment variables to set :
 - CLUE_INPUT_FILE : full path to CLUE_clusters.root 
 - PORT, HOST, DASH_DEBUG : for Dash
"""
import urllib.parse
import os

import dash
from dash import Dash, html, Input, Output, State, dcc
from dash.exceptions import PreventUpdate
import flask_caching
import uproot
import awkward as ak

from hists.parameters import beamEnergies, ntupleNumbersPerBeamEnergy
from event_visualizer_plotly.utils import EventLoader, EventID, LoadedEvent
from event_visualizer_plotly.vis_clue3D import Clue3DVisualization
from event_visualizer_plotly.vis_layer import LayerVisualization
from event_visualizer_plotly.vis_longitudinal_profile import LongitudinalProfileVisualization

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog="dash_event_visualizer",
        description="CLUE3D event visualizer using Dash",
    )
    parser.add_argument('-p', '--port', default=8051, help="Port to listen on")
    parser.add_argument('-i', '--input-file', default="/eos/user/t/tcuisset/hgcal/testbeam18-clue3d/v33/cmssw/data/CLUE_clusters.root",
        dest="input_file", help="Path to CLUE_clusters.root file (included)")
    parser.add_argument('-d', '--debug', action=argparse.BooleanOptionalAction, help="Enable debug mode", dest="debug", default=True)
    parser.add_argument('-H', '--host', dest="host", default=None,
        help="Host name to bin on (if not specified, use Dash default  which is env variable HOST or 127.0.0.1). On llruicms you should put 'llruicms01'")
    args = parser.parse_args()
    clueInputFile = args.input_file
else:
    clueInputFile = os.environ["CLUE_INPUT_FILE"]

# Local : /data_cms_upgrade/cuisset/testbeam18/clue3d/v33/cmssw/data/CLUE_clusters.root
eventLoader = EventLoader(clueInputFile)

app = Dash(__name__)
server = app.server

cache = flask_caching.Cache(app.server, config={
    "CACHE_TYPE":"FileSystemCache",
    "CACHE_DIR":"./.cache"
})

legendDivStyle = {'flex': '0 1 auto', 'margin':"10px"}
app.layout = html.Div([
    html.Div([
        dcc.Location(id="url", refresh=False), # For some reason  "callback-nav" works but False does not
        dcc.Store(id="signal-event-ready"),

        html.H1(children='CLUE3D event visualizer (right-click to rotate, left-click to move)'),
        html.Div(children=[
            html.Div("Beam energy (GeV) :", style=legendDivStyle),
            dcc.Dropdown(options=beamEnergies, value=20, id="beamEnergy", style={'flex': '1 1 auto'}),
            html.Div("Ntuple number :", style=legendDivStyle),
            dcc.Dropdown(id="ntupleNumber", style={'flex': '1 1 auto'}),
            html.Div("Event :", style=legendDivStyle),
            dcc.Dropdown(id="event", style={'flex': '1 1 auto'}),
            html.Div("Layer (for layer view) :"),
            html.Div(dcc.Slider(min=1, max=28, step=1, value=10, id="layer"), style={"flex":"10 10 auto"}),
            dcc.Clipboard(id="link-copy-clipboard", title="Copy link", content="abc"),
        ], style={"display":"flex", "flex-flow":"row"}),
    ], style={'flex': '0 1 auto'}),
    dcc.Tabs(id="plot_tabs", children=[
        dcc.Tab(label="3D view", value="3D", children=[
            dcc.Graph(id="plot_3D", style={"height":"100%"}, config=dict(toImageButtonOptions=dict(
                scale=3.
            ))),
        ]),
        dcc.Tab(label="Layer view", value="layer", children=[
            dcc.Graph(id="plot_layer", style={"height":"100%"}, config=dict(toImageButtonOptions=dict(
                scale=4.
            ))),
        ]),
        dcc.Tab(label="Longitudinal profile", value="longitudinal_profile", children=[
            dcc.Graph(id="plot_longitudinal-profile", style={"height":"100%"})
        ]),
    ], parent_style={'flex': '1 1 auto'}, content_style={'flex': '1 1 auto'}, value="3D")
    
], style={'display': 'flex', 'flex-flow': 'column', "height":"100vh"})

@app.callback(
    Output("ntupleNumber", "options"),
    [Input("beamEnergy", "value")]
)
def update_ntupleNumber(beamEnergy):
    return ntupleNumbersPerBeamEnergy[int(beamEnergy)]

@app.callback(
    Output("event", "options"),
    [Input("ntupleNumber", "value")]
)
def update_available_events(ntupleNumber):
    return eventLoader.tree.arrays(cut=f"(ntupleNumber == {ntupleNumber})", filter_name=["event"]).event

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


@app.callback(
    [Output("beamEnergy", "value"), Output("ntupleNumber", "value"), Output("event", "value"), Output("layer", "value"), Output("plot_tabs", "value")],
    [Input("url", "search")]
)
def simpleUrlUpdate(urlSearchValue):
    """ On initial load, set settings from URL"""
    print("simpleUrlUpdate", dash.ctx.triggered_prop_ids, dash.ctx.inputs, flush=True)
    try:
        parsed_url_query = urllib.parse.parse_qs(urlSearchValue[1:]) # Drop the leading "?"
        try:
            plot_tabs_value = parsed_url_query["tab"][0]
        except:
            plot_tabs_value = dash.no_update
        try:
            layer_value = int(parsed_url_query["layer"][0])
        except:
            layer_value = dash.no_update

        return int(parsed_url_query["beamEnergy"][0]), int(parsed_url_query["ntuple"][0]), int(parsed_url_query["event"][0]), layer_value, plot_tabs_value
    except KeyError:
        raise PreventUpdate()

@app.callback(
    Output("link-copy-clipboard", "content"),
    [   Input("link-copy-clipboard", "n_clicks"), State("url", "href"),
        State("beamEnergy", "value"), State("ntupleNumber", "value"), State("event", "value"), State("layer", "value"), State("plot_tabs", "value")
    ]
)
def updateLinkClipboard(_, href, beamEnergy, ntuple, event, layer, plot_tab):
    url_tuple = urllib.parse.urlparse(href)
    url_query = dict(beamEnergy=beamEnergy, ntuple=ntuple, event=event, layer=layer, tab=plot_tab)
    new_url_tuple = url_tuple._replace(query=urllib.parse.urlencode(url_query)) # namedtuple are immutable
    return urllib.parse.urlunparse(new_url_tuple)

@cache.memoize(timeout=60*5) # cache for 5 minutes 
def loadEvent(ntuple, eventNb) -> LoadedEvent:
    if ntuple is not None and eventNb is not None:
        return eventLoader.loadEvent(EventID(ntuple, eventNb))
    raise RuntimeError()

@app.callback(
    Output("signal-event-ready", "data"),
    [Input("ntupleNumber", "value"), Input("event", "value")],
)
def loadEventCallback(ntuple, event):
    print("loadEventCallback", dash.ctx.triggered_prop_ids, dash.ctx.inputs, flush=True)
    if ntuple is not None and event is not None:
        loadEvent(ntuple, event) # just load the cache, discard results
    return dict(ntupleNumber=ntuple, event=event)

@app.callback(
    [Output("plot_3D", "figure"), Output("plot_longitudinal-profile", "figure")],
    [Input("signal-event-ready", "data")]
)
def mainEventUpdate(storeData):
    """ Main callback to update all the plots at the same time.
    layer is State as there is another callback updateOnlyLayerPlot to update just the layer view """
    if storeData is None:
        return None, None
    eventNb, ntupleNumber = storeData["event"], storeData["ntupleNumber"]

    print("mainEventUpdate", dash.ctx.triggered_prop_ids, dash.ctx.inputs, flush=True)
    print(flush=True)
    if eventNb is None or ntupleNumber is None:
        return None, None
    try:
        event = loadEvent(ntupleNumber, eventNb)
        plot_3D = makePlotClue3D(event)
        plot_longitudinal = makePlotLongitudinalProfile(event)
    except RuntimeError:
        plot_3D = None
        plot_longitudinal = None
    
    return plot_3D, plot_longitudinal


@app.callback(
    Output("plot_layer", "figure"),
    [Input("signal-event-ready", "data"), Input("layer", "value")],
    prevent_initial_call=True, # On intial loading the layer view is loaded by the main callback mainEventUpdate
)
def updateOnlyLayerPlot(storeData, layer):
    """ Small callback to only update the layer view """
    eventNb, ntupleNumber = storeData["event"], storeData["ntupleNumber"]
    try:
        event = loadEvent(ntupleNumber, eventNb)
    except RuntimeError:
        return None
    return makePlotLayer(event, layer)

if __name__ == '__main__':
    run_kwargs = {"debug": args.debug, "port":args.port}
    if args.host is not None:
        run_kwargs["host"] = args.host
    #if args.debug:
    #    run_kwargs["threaded"] = False # For easier debugging
    print(run_kwargs)
    app.run(**run_kwargs)
""" Event visualizer using Dash
Environment variables to set :
 - CLUE_INPUT_FILE : full path to CLUE_clusters.root 
 - PORT, HOST, DASH_DEBUG : for Dash
"""
import urllib.parse
import os
import glob
import collections

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
    # Local : /data_cms_upgrade/cuisset/testbeam18/clue3d/v33/cmssw/data/CLUE_clusters.root
    # EOS : "/eos/user/t/tcuisset/hgcal/testbeam18-clue3d/v33/cmssw/data/CLUE_clusters.root",
    parser.add_argument('-i', '--input-file', default=None, 
        dest="input_file", help="Path to CLUE_clusters.root file, in single-file mode (specify either -i or -I, not both)")
    # EOS : "/eos/user/t/tcuisset/hgcal/testbeam18-clue3d/v33/"
    # local : /data_cms_upgrade/cuisset/testbeam18/clue3d/v33/
    parser.add_argument('-I', '--input-folder', default="/data_cms_upgrade/cuisset/testbeam18/clue3d/v33/",
        dest="input_folder", help="Path to folders holding CLUE_clusters.root files. Beyond this path, subdirectories should exist in the form *clue-params*/*datatype*/CLUE_clusters.root. Incompatible with -i")
    parser.add_argument('-d', '--debug', action=argparse.BooleanOptionalAction, help="Enable debug mode", dest="debug", default=True)
    parser.add_argument('-H', '--host', dest="host", default=None,
        help="Host name to bin on (if not specified, use Dash default  which is env variable HOST or 127.0.0.1). On llruicms you should put 'llruicms01'")
    args = parser.parse_args()
    clueInputFile = args.input_file
    clueInputFolder = args.input_folder
else:
    try:
        clueInputFile = os.environ["CLUE_INPUT_FILE"]
    except KeyError:
        clueInputFile = None
    try:
        clueInputFolder = os.environ["CLUE_INPUT_FOLDER"]
    except KeyError:
        clueInputFolder = None

if (clueInputFile is None and clueInputFolder is None) or (clueInputFile is not None and clueInputFolder is not None):
    raise ValueError("You should specify either -i or -I (or set either CLUE_INPUT_FILE or CLUE_INPUT_FOLDER environment variables)")

# Discover clue-params and datatypes
availableSamples = collections.defaultdict(dict) # dict clueParam -> dict : datatype -> EventLoader
if clueInputFolder is not None:
    curdir = os.getcwd()
    os.chdir(clueInputFolder) # using glob root_dir option is only available from Python 3.10
    paths = glob.glob(os.path.join('*',      '*',  'CLUE_clusters.root')) # , root_dir=clueInputFolder
    os.chdir(curdir)
    for path in paths:
        folderPath, _ = os.path.split(path) # folderPath = clueParams/datatype, _=CLUE_clusters.root
        clueParam, datatype = os.path.split(folderPath)
        availableSamples[clueParam][datatype] = EventLoader(os.path.join(clueInputFolder, path))
else:
    availableSamples["N/A"]["N/A"] = EventLoader(clueInputFile)
availableSamples:dict[str, dict[str, EventLoader]] = dict(availableSamples) # transform from defaultdict to dict to avoid accidentally adding keys

clueParamsList = list(availableSamples.keys())
try: # Put "cmssw" as first list element
    clueParamsList.insert(0, clueParamsList.pop(clueParamsList.index("cmssw")))
except ValueError:
    pass

app = Dash(__name__)
server = app.server

cache = flask_caching.Cache(app.server, config={
    "CACHE_TYPE":"FileSystemCache",
    "CACHE_DIR":"./.cache"
})

legendDivStyle = {'flex': '0 1 auto', 'margin':"10px"}
dropdownStyle = {'flex': '1 1 auto'}
app.layout = html.Div([ # Outer Div
    html.Div([
        dcc.Location(id="url", refresh=False), # For some reason  "callback-nav" works but False does not
        dcc.Store(id="signal-event-ready"),

        html.H1(children='CLUE3D event visualizer (right-click to rotate, left-click to move)'),
        html.Div(children=[
            html.Div("ClueParams :", style=legendDivStyle),
            dcc.Dropdown(options=clueParamsList, id="clueParam", style=dropdownStyle),
            dcc.Dropdown(id="datatype", style={"width":"200px"}),
            html.Div("Beam energy (GeV) :", style=legendDivStyle),
            dcc.Dropdown(options=beamEnergies, value=20, id="beamEnergy", style=dropdownStyle),
            html.Div("Ntuple number :", style=legendDivStyle),
            dcc.Dropdown(id="ntupleNumber", style=dropdownStyle),
            html.Div("Event :", style=legendDivStyle),
            dcc.Dropdown(id="event", style=dropdownStyle),
            html.Div("Layer (for layer view) :"),
            html.Div(dcc.Slider(min=1, max=28, step=1, value=10, id="layer"), style={"flex":"10 10 auto"}),
            dcc.Clipboard(id="link-copy-clipboard", title="Copy link", content="abc"),
        ], style={"display":"flex", "flexFlow":"row"}),
    ], style={'flex': '0 1 auto'}),
    dcc.Loading(
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
        ], parent_style={"height":"100%"}, # Use whole height of Loading div (not sure why this is needed, but without it it does not work)
        content_style={'flex': '1 1 auto'}, # Have tab content flex vertically inside [tab header, tab content] div
        value="3D"),
    parent_style={'flex': '1 1 auto'}
    ),
    
], style={'display': 'flex', 'flexFlow': 'column', "height":"100vh"})

class FullEventID(collections.namedtuple("FullEventID", ["clueParam", "datatype", "beamEnergy", "ntupleNumber", "event"], defaults=[None]*5)):
    @classmethod
    def fromDict(cls, d:dict) -> "FullEventID":
        return cls(**{key : d.get(key, cls._field_defaults[key]) for key in cls._fields})
    
    @classmethod
    def fromCallbackContext(cls, args_grouping:dict) -> "FullEventID":
        dictRes = dict()
        for arg_dict in args_grouping:
            # arg_dict is a dict with id->.., value->..., etc
            if arg_dict.id in cls._fields:
                dictRes[arg_dict.id] = arg_dict.value
        return cls(**dictRes)

    def toEventId(self) -> EventID:
        return EventID(self.beamEnergy, self.ntupleNumber, self.event)
    
    def isFilled(self):
        return not any(map(lambda x: x is None, self))

@app.callback(
    Output("datatype", "options"),
    [Input("clueParam", "value")]
)
def update_datatypes(clueParam):
    print("update_datatypes", dash.ctx.triggered_prop_ids, dash.ctx.inputs, flush=True)
    try:
        datatypes = list(availableSamples[clueParam].keys())
        try: # Put "data" as first list element
            datatypes.insert(0, datatypes.pop(datatypes.index("data")))
        except ValueError:
            pass
        return datatypes
    except:
        return []

@app.callback(
    Output("ntupleNumber", "options"),
    [Input("clueParam", "value"), Input("datatype", "value"), Input("beamEnergy", "value")]
)
def update_ntupleNumber(clueParam, datatype, beamEnergy):
    print("update_ntupleNumber", dash.ctx.triggered_prop_ids, dash.ctx.inputs, flush=True)
    try:
        ntuple_energy_pairs = availableSamples[clueParam][datatype].ntuplesEnergies
        return list(ntuple_energy_pairs.ntupleNumber[ntuple_energy_pairs.beamEnergy == beamEnergy])
    except:
        return []

@app.callback(
    Output("event", "options"),
    [Input("clueParam", "value"), Input("datatype", "value"), Input("beamEnergy", "value"), Input("ntupleNumber", "value")]
)
def update_availableEvents(clueParam, datatype, beamEnergy, ntupleNumber):
    print("update_availableEvents", dash.ctx.triggered_prop_ids, dash.ctx.inputs, flush=True)
    try:
        return list(availableSamples[clueParam][datatype].eventNumbersPerNtuple(beamEnergy, ntupleNumber))
    except:
        return []

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
    [Output("clueParam", "value"), Output("datatype", "value"), Output("beamEnergy", "value"), Output("ntupleNumber", "value"), Output("event", "value"), Output("layer", "value"), Output("plot_tabs", "value")],
    [Input("url", "search")]
)
def simpleUrlUpdate(urlSearchValue):
    """ On initial load, set settings from URL"""
    if not dash.ctx.triggered:
        # not initial load : ignore
        raise PreventUpdate()
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

        return parsed_url_query["clueParam"][0], parsed_url_query["datatype"][0], int(parsed_url_query["beamEnergy"][0]), int(parsed_url_query["ntuple"][0]), int(parsed_url_query["event"][0]), layer_value, plot_tabs_value
    except (KeyError, ValueError) as e: # catch non-existing key or catch None
        print("simpleUrlUpdate : failed parse")
        print(e)
        raise PreventUpdate()

@app.callback(
    Output("link-copy-clipboard", "content"),
    [   Input("link-copy-clipboard", "n_clicks"), State("url", "href"),
        State("clueParam", "value"), State("datatype", "value"), State("beamEnergy", "value"), State("ntupleNumber", "value"), State("event", "value"), State("layer", "value"), State("plot_tabs", "value")
    ]
)
def updateLinkClipboard(_, href, clueParam, datatype, beamEnergy, ntuple, event, layer, plot_tab):
    url_tuple = urllib.parse.urlparse(href)
    url_query = dict(clueParam=clueParam, datatype=datatype, beamEnergy=beamEnergy, ntuple=ntuple, event=event, layer=layer, tab=plot_tab)
    new_url_tuple = url_tuple._replace(query=urllib.parse.urlencode(url_query)) # namedtuple are immutable
    return urllib.parse.urlunparse(new_url_tuple)

@cache.memoize(timeout=60*5) # cache for 5 minutes 
def loadEvent(fullEventId:FullEventID) -> LoadedEvent:
    return availableSamples[fullEventId.clueParam][fullEventId.datatype].loadEvent(fullEventId.toEventId())

@app.callback(
    Output("signal-event-ready", "data"),
    [State("clueParam", "value"), State("datatype", "value"), State("beamEnergy", "value"), State("ntupleNumber", "value"), Input("event", "value")],
)
def loadEventCallback(clueParam, datatype, beamEnergy, ntuple, event):
    print("loadEventCallback", dash.ctx.triggered_prop_ids, dash.ctx.inputs)
    fullEventID = FullEventID.fromCallbackContext(dash.ctx.args_grouping)
    #print(dash.ctx.args_grouping)
    print(fullEventID)
    if fullEventID.isFilled():
        
        loadEvent(fullEventID) # just load the cache, discard results
    return fullEventID._asdict()

def isStorageValid(storage_eventId:dict) -> bool:
    return (storage_eventId is not None) and len(storage_eventId) > 0

@app.callback(
    [Output("plot_3D", "figure"), Output("plot_longitudinal-profile", "figure")],
    [Input("signal-event-ready", "data")]
)
def mainEventUpdate(storage_eventId):
    """ Main callback to update all the plots at the same time.
    layer is State as there is another callback updateOnlyLayerPlot to update just the layer view """
    print("mainEventUpdate", dash.ctx.triggered_prop_ids, dash.ctx.inputs, flush=True)
    if not isStorageValid(storage_eventId):
        return {}, {}
    fullEventID = FullEventID(**storage_eventId)
    if not fullEventID.isFilled():
        return {}, {}
    
    try:
        event = loadEvent(fullEventID)
        plot_3D = makePlotClue3D(event)
        plot_longitudinal = makePlotLongitudinalProfile(event)
    except:
        plot_3D = {}
        plot_longitudinal = {}
    
    return plot_3D, plot_longitudinal


@app.callback(
    Output("plot_layer", "figure"),
    [Input("signal-event-ready", "data"), Input("layer", "value")],
    prevent_initial_call=True, # On intial loading the layer view is loaded by the main callback mainEventUpdate
)
def updateOnlyLayerPlot(storage_eventId, layer):
    """ Small callback to only update the layer view """
    if not isStorageValid(storage_eventId):
        return {}
    fullEventID = FullEventID(**storage_eventId)
    if not fullEventID.isFilled():
        return {}
    try:
        event = loadEvent(fullEventID)
    except:
        return {}
    return makePlotLayer(event, layer)

if __name__ == '__main__':
    run_kwargs = {"debug": args.debug, "port":args.port}
    if args.host is not None:
        run_kwargs["host"] = args.host
    #if args.debug:
    #    run_kwargs["threaded"] = False # For easier debugging
    print(run_kwargs)
    app.run(**run_kwargs)
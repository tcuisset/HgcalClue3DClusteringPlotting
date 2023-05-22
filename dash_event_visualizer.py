""" Event visualizer using Dash
Environment variables to set :
 - CLUE_INPUT_FILE : full path to CLUE_clusters.root 
 - PORT, HOST, DASH_DEBUG : for Dash
"""
import urllib.parse
import os
import sys
import glob
import collections
import traceback
from timeit import default_timer as timer

import dash
from dash import Dash, html, Input, Output, State, dcc
from dash.exceptions import PreventUpdate
import flask_caching
import uproot
import awkward as ak

from hists.parameters import beamEnergies, ntupleNumbersPerBeamEnergy
from event_visualizer.event_index import EventLoader, EventID, LoadedEvent
from event_visualizer.dash_app.plots import makePlotClue3D, zAxisDropdownSettings, makePlotLayer, makePlotLongitudinalProfile
from event_visualizer.dash_app.tables import makeClus2DTable, makeClus3DTable, updateClus2DTableData, updateClus3DTableData
from event_visualizer.dash_app.views import view_3D_component, view_layer_component
from event_visualizer.dash_app.event_selection_bar import makeEventSelectionBarComponent, registerEventSelectionBarCallbacks

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

print("Loading samples...")
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

print("Done")

app = Dash(__name__)
server = app.server

cache = flask_caching.Cache(app.server, config={
    "CACHE_TYPE":"FileSystemCache",
    "CACHE_DIR":"./.cache"
})

eventVisInstructions = dcc.Markdown('''
# CLUE3D event visualizer
## Selecting an event
Fill in all dropdowns from left to right to select an event

clueParams : settings of CLUE and CLUE3D algorithms (cmssw is nearly the same parameters as in CMSSW)

## Using the 3D view
### Moving around
Hold right-click and drag to rotate

Hold left-click and drag to move

### Legend
You can click on legend elements (in right menu) to show or hide them.
For example, you can click on "Rechits" and on "Rechits chain" to hide all rechits-related information ("Rechits chain" selects the arrows of nearest higher chain from CLUE)

### Aspect ratio and perspective
The first dropdown below the tabs controls the z axis : you can have either layer number or z position.
The "aspect ratio" setting increases the z axis factor for better visibility in the 3D view.
The second dropdown changes the projection type : projective (normal projection) or orthographic (parallels stay parallels).

## Layer view
Select the layer using the slider at top right

## Longitudinal profile
Histogram of energy distribution per layer in the current event
Overlaid are diistribution for rechits (blue) and CLUE (red, taking only rechits that are members of a layer cluster)

## Save a link to a specific event
Click the clipboard button at the very top right, it will save a direct link to the current event in the clipboard
''')

app.layout = html.Div([ # Outer Div
    dcc.Location(id="url", refresh=False), # For some reason  "callback-nav" works but False does not
    dcc.Store(id="signal-event-ready"),
    html.Div([makeEventSelectionBarComponent(clueParamsList, beamEnergies)
    ], style={'flex': '0 1 auto'}),
    
    dcc.Tabs(id="plot_tabs", children=[
        dcc.Tab(label="3D view", value="3D", children=view_3D_component),
        dcc.Tab(label="Layer view", value="layer", children=view_layer_component),
        dcc.Tab(label="Longitudinal profile", value="longitudinal_profile", children=
            dcc.Loading(
                children=dcc.Graph(id="plot_longitudinal-profile", style={"height":"100%"}),
                parent_style={"flex": "1 1 auto"}, # graph should spread vertically as much as possible (note there is only one box in the flex box)
            )
        ),
        dcc.Tab(label="Tables", value="clus3D_table", children=[
            html.H3("Tracksters"),
            makeClus3DTable(),
            html.H3("Layer clusters"),
            makeClus2DTable(),
        ]),
        dcc.Tab(label="Instructions", children=eventVisInstructions, value="instructions"),
    ],
    parent_style={'flex': '1 1 auto'}, # Have the the whole tabs Div flex vertically inside outer div
    content_style={'flex': '1 1 auto',  # Have tab content flex vertically inside [tab header, tab content] div
                   # Have stuff arranged vertically inside a tab (only actually needed for 3D view tab, but I have not found a way to have per-tab setting)
                   "display":"flex", "flexFlow":"column"},
    value="instructions" # Open instructions tab by default (unless overriden by url)
    ),
    
], 
style={'display': 'flex', 'flexFlow': 'column', # Have main button row and tabs outer div arranged vertically
       "height":"100vh"}) # Always take whole viewport height


registerEventSelectionBarCallbacks(availableSamples)

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
            plot_tabs_value = "3D" # open 3D view by default
        try:
            layer_value = int(parsed_url_query["layer"][0])
        except:
            layer_value = dash.no_update

        return parsed_url_query["clueParam"][0], parsed_url_query["datatype"][0], int(parsed_url_query["beamEnergy"][0]), int(parsed_url_query["ntuple"][0]), int(parsed_url_query["event"][0]), layer_value, plot_tabs_value
    except (KeyError, ValueError) as e: # catch non-existing key or catch None
        print("simpleUrlUpdate : failed parse", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
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


emptyFigure = { "data": [], "layout": {}, "frames": [],}
emptyTable = []

@app.callback(
    [Output("plot_3D", "figure"), Output("plot_longitudinal-profile", "figure"), 
     Output("clus3D_table", "data"), Output("clus2D_table", "data"), ],
    [Input("signal-event-ready", "data"), Input("zAxisSetting", "value"), Input("projectionType", "value")]
)
def mainEventUpdate(storage_eventId, zAxisSetting, projectionType):
    """ Main callback to update all the plots at the same time.
    layer is State as there is another callback updateOnlyLayerPlot to update just the layer view """
    print("mainEventUpdate", dash.ctx.triggered_prop_ids, dash.ctx.inputs, flush=True)
    emptyReturn = emptyFigure, emptyFigure, emptyTable, emptyTable

    if not isStorageValid(storage_eventId):
        return emptyReturn
    fullEventID = FullEventID(**storage_eventId)
    if not fullEventID.isFilled():
        return emptyReturn
    
    try:
        start_time = timer()
        event = loadEvent(fullEventID)
        dash.callback_context.record_timing('loadedEvent', timer() - start_time, 'Loading the event')
        return makePlotClue3D(event, zAxisSetting, projectionType), makePlotLongitudinalProfile(event), updateClus3DTableData(event), updateClus2DTableData(event)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return emptyReturn


@app.callback(
    Output("plot_layer", "figure"),
    [Input("signal-event-ready", "data"), Input("layer", "value")],
    prevent_initial_call=True, # On intial loading the layer view is loaded by the main callback mainEventUpdate
)
def updateOnlyLayerPlot(storage_eventId, layer):
    """ Small callback to only update the layer view """
    if not isStorageValid(storage_eventId):
        return emptyFigure
    fullEventID = FullEventID(**storage_eventId)
    if not fullEventID.isFilled():
        return emptyFigure
    try:
        start_time = timer()
        event = loadEvent(fullEventID)
        dash.callback_context.record_timing('loadedEvent', timer() - start_time, 'Loading the event')
        return makePlotLayer(event, layer)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return emptyFigure

if __name__ == '__main__':
    run_kwargs = {"debug": args.debug, "port":args.port}
    if args.host is not None:
        run_kwargs["host"] = args.host
    #if args.debug:
    #    run_kwargs["threaded"] = False # For easier debugging
    print(run_kwargs)
    app.run(**run_kwargs)
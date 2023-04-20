import argparse

from dash import Dash, html, Input, Output, dcc
import uproot
import awkward as ak

from hists.parameters import beamEnergies, ntupleNumbersPerBeamEnergy
from event_visualizer_plotly.utils import EventLoader, EventID
from event_visualizer_plotly.vis_clue3D import Clue3DVisualization
from event_visualizer_plotly.vis_layer import LayerVisualization

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

@app.callback(
    Output("plot_3D", "figure"),
    [Input("ntupleNumber", "value"), Input("event", "value")]
)
def update_plot3D(ntupleNumber, eventNb):
    event = eventLoader.loadEvent(EventID(ntupleNumber, eventNb))
    fig = (Clue3DVisualization(event)
        .add3DClusters()
        .add2DClusters()
        .addRechits()
        .addImpactTrajectory()
    ).fig
    fig.update_layout(dict(uirevision=1)) # Keep the current view in any case. See https://community.plotly.com/t/preserving-ui-state-like-zoom-in-dcc-graph-with-uirevision-with-dash/15793
    return fig


@app.callback(
    Output("plot_layer", "figure"),
    [Input("ntupleNumber", "value"), Input("event", "value"), Input("layer", "value")]
)
def update_plot3D(ntupleNumber, eventNb, layer):
    event = eventLoader.loadEvent(EventID(ntupleNumber, eventNb))
    fig = (LayerVisualization(event, layerNb=layer)
        .add2DClusters()
        .addRechits()
        .addImpactPoint()
        .addCircleSearchForComputingClusterPosition()
    ).fig
    fig.update_layout(dict(uirevision=1))
    return fig

if __name__ == '__main__':
    if args.host is None:
        app.run(debug=args.debug, port=args.port)
    else:
        app.run(debug=args.debug, port=args.port, host=args.host)
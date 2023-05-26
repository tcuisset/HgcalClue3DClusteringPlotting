import random
import pandas as pd
import awkward as ak
import dash
from dash import html, dcc, Input, Output, State
from jupyter_dash import JupyterDash

from event_visualizer.plotter.layer import LayerVisualization
from event_visualizer.plotter.clue3D import Clue3DVisualization
from event_visualizer.event_index import EventLoader, EventID, LoadedEvent
from event_visualizer.dash_app.views import view_3D_component, view_layer_component
from event_visualizer.dash_app.plots import makePlotClue3D, makePlotLayer, makePlotLongitudinalProfile
from locateEvents.utils import makeDashLink, makeCsvRow, printCsvRowsFromDf


class EventDisplay:
    def __init__(self, eventList:pd.DataFrame|ak.Array|dict, eventLoader:EventLoader, run_server_mode="inline") -> None:
        """ Build notebook-embedded event display
        Parameters : 
         - eventList : should contain beamEnergy, event, ntupleNumber columns, either as a pandas Dataframe, an awkard array or a dict of lists
         - eventLoader : the EventLoader to load the events from
         - run_server_mode : passed to JupyterDash.run_server, can be "external", "inline", or "jupyter_lab"
        """
        if isinstance(eventList, ak.Array):
            eventList = ak.to_dataframe(eventList[["beamEnergy", "event", "ntupleNumber"]])
        if isinstance(eventList, dict):
            eventList = pd.DataFrame(eventList)
        self.eventList = eventList
        self.el = eventLoader
        self.shuffledIndex = None

        self.app = JupyterDash(__name__)
        self.app.layout = \
        html.Div(children=[
            dcc.Store(id="current-event-id"),
            html.Div(children=[
                html.Button(id="button-next", children="Next"),
            ], style={"flex":"0 1 auto"}),
            dcc.Tabs(id="plot_tabs", children=[
                dcc.Tab(label="3D view", value="3D", children=view_3D_component),
                dcc.Tab(label="Layer view", value="layer", children=view_layer_component),
                dcc.Tab(label="Longitudinal profile", value="longitudinal_profile", children=
                    dcc.Loading(
                        children=dcc.Graph(id="plot_longitudinal-profile", style={"height":"100%"}),
                        parent_style={"flex": "1 1 auto"}, # graph should spread vertically as much as possible (note there is only one box in the flex box)
                    )
                ),
            ], style={"flex":"1 1 auto"}, value="3D"),
        ], style={"display":"flex", "flexFlow":"column"})
        

        @self.app.callback(
            [Output("plot_3D", "figure"), Output("plot_longitudinal-profile", "figure")],
            [Input("current-event-id", "data"), Input("zAxisSetting", "value"), Input("projectionType", "value")],
        )
        def updatePlots(eventId, zAxisSetting, projectionType):            
            return makePlotClue3D(self.currentEvent, zAxisSetting, projectionType), makePlotLongitudinalProfile(self.currentEvent)

        @self.app.callback(
            Output("plot_layer", "figure"),
            [Input("current-event-id", "data"), Input("layer", "value")],
        )
        def updateLayerPlot(eventId, layer):
            return makePlotLayer(self.currentEvent, layer)
        

        @self.app.callback(
            [Output("current-event-id", "data"), Output("layer", "value")],
            [Input("button-next", "n_clicks")],
        )
        def updateEvent(_):
            index = self.sampleRandom()
            self.currentEvent, layer = self.loadEvent(index)
            if layer is not None:
                return index, layer
            else:
                return index, dash.no_update

        self.currentEvent = None

        self.app.run_server(run_server_mode, debug=True)


    
    def loadEvent(self, index) -> tuple[LoadedEvent, int]:
        """ Returns LoadedEvent, layerNb tuple. index is integer index starting from 0 into df (uses iloc) """
        if "layer" in self.eventList:
            layer = self.eventList.iloc[index]
        else:
            layer = None
        return self.el.loadEvent(EventID(self.eventList.beamEnergy.loc[index], self.eventList.ntupleNumber.loc[index], self.eventList.event.loc[index])), layer
    
    def sampleRandom(self):
        if self.shuffledIndex is None:
            self.shuffledIndex = self.eventList.index.to_list()
            random.shuffle(self.shuffledIndex)
            self.shuffledIndexIter = iter(self.shuffledIndex)
        
        return next(self.shuffledIndexIter)
import random
import pandas as pd
import awkward as ak
import dash
from dash import html, dcc, Input, Output, State
#from jupyter_dash import JupyterDash

from event_visualizer.plotter.layer import LayerVisualization
from event_visualizer.plotter.clue3D import Clue3DVisualization
from event_visualizer.event_index import EventLoader, EventID, LoadedEvent
from event_visualizer.dash_app.views import view_3D_component, view_layer_component
from event_visualizer.dash_app.plots import makePlotClue3D, makePlotLayer, makePlotLongitudinalProfile
from locateEvents.utils import makeDashLink, makeCsvRow, printCsvRowsFromDf


class EventDisplay:
    def __init__(self, eventList:pd.DataFrame|ak.Array|dict, eventLoader:EventLoader, jupyter_mode="inline", run_server_kwargs=dict(), clue3DPlotSettings=dict()) -> None:
        """ Build notebook-embedded event display
        Parameters : 
         - eventList : should contain beamEnergy, event, ntupleNumber columns, either as a pandas Dataframe, an awkard array or a dict of lists
         - eventLoader : the EventLoader to load the events from
         - jupyter_mode : passed to Dash.run, can be "external", "inline", or "jupyter_lab"
         - extra kwargs : passed to Dash.run
        """
        if isinstance(eventList, ak.Array):
            eventList = ak.to_dataframe(eventList[["beamEnergy", "event", "ntupleNumber"]])
        if isinstance(eventList, dict):
            eventList = pd.DataFrame(eventList)
        self.eventList = eventList
        self.el = eventLoader
        self.shuffledIndex = None

        self.app = dash.Dash(__name__)
        self.app.layout = \
        html.Div(children=[
            dcc.Store(id="current-event-id"),
            html.Div(children=[
                html.Button(id="button-next", children="Next"),
                html.Div(id="relayout-data-output"),
            ], style={"flex":"0 1 auto", "flexFlow":"row"}),
            dcc.Tabs(id="plot_tabs", children=[
                dcc.Tab(label="3D view", value="3D", children=view_3D_component),
                dcc.Tab(label="Layer view", value="layer", children=view_layer_component),
                dcc.Tab(label="Longitudinal profile", value="longitudinal_profile", children=
                    dcc.Loading(
                        children=dcc.Graph(id="plot_longitudinal-profile", style={"height":"100%"}),
                        parent_style={"flex": "1 1 auto"}, # graph should spread vertically as much as possible (note there is only one box in the flex box)
                    )
                ),
            ], style={"flex":"1 0 auto"}, value="3D", parent_style={'flex': '1 1 auto'},
                content_style={'flex': '1 1 auto',  # Have tab content flex vertically inside [tab header, tab content] div
                   # Have stuff arranged vertically inside a tab (only actually needed for 3D view tab, but I have not found a way to have per-tab setting)
                   "display":"flex", "flexFlow":"column"},),
        ], style={"display":"flex", "flexFlow":"column", "height":"100vh"})
        

        @self.app.callback(
            [Output("plot_3D", "figure"), Output("plot_longitudinal-profile", "figure")],
            [Input("current-event-id", "data"), Input("zAxisSetting", "value"), Input("projectionType", "value")],
        )
        def updatePlots(eventId, zAxisSetting, projectionType):            
            fig_clue3d, fig_longProf =  makePlotClue3D(self.currentEvent, zAxisSetting, projectionType, visSettings=clue3DPlotSettings), makePlotLongitudinalProfile(self.currentEvent)
            fig_clue3d.update_scenes(
                xaxis_visible=False, yaxis_visible=False,zaxis_visible=False 
                )
            return fig_clue3d, fig_longProf

        @self.app.callback(
            Output("plot_layer", "figure"),
            [Input("current-event-id", "data"), Input("layer", "value")],
        )
        def updateLayerPlot(eventId, layer):
            fig = makePlotLayer(self.currentEvent, layer)
            fig.update_layout(
                xaxis_visible=False, yaxis_visible=False,
                xaxis_showgrid=False, yaxis_showgrid=False, paper_bgcolor='rgba(0,0,0,0)',  plot_bgcolor='rgba(0,0,0,0)'
                )
            return fig
        
        # @self.app.callback(
        #     Output("relayout-data-output", "children"),
        #     Input("plot_3D", "relayoutData")
        # )
        # def show_relayout_data(data):
        #     return [str(data)]
        
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

        self.app.run(jupyter_mode=jupyter_mode, debug=True, **run_server_kwargs)


    
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
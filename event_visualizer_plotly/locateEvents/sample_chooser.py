import pandas as pd
from IPython.display import display, clear_output
import ipywidgets as widgets
from plotly.io.base_renderers import PlotlyRenderer, NotebookRenderer
import plotly.io as pio

from event_visualizer_plotly.utils import EventLoader, EventID
from event_visualizer_plotly.vis_clue3D import Clue3DVisualization

class EventDisplayList:
    def __init__(self, eventList:list[EventID]) -> None:
        next = widgets.Button(description="Next")
        prev = widgets.Button(description="Previous")
        self.button_container = widgets.HBox([next, prev])
        next.on_click(self.next)
        prev.on_click(self.prev)
        self.current_i = 0
        self.eventList = eventList
        self.el = EventLoader('/data_cms_upgrade/cuisset/testbeam18/clue3d/v31/cmssw/data/CLUE_clusters.root')

        #display(self.button_container)

    def next(self, b):
        if self.current_i+1 < len(self.eventList):
            self.current_i += 1
            self.showPlot()
    def prev(self, b):
        if self.current_i-1 >= 0:
            self.current_i -= 1
            self.showPlot()
        
    def mainCell(self):
        self.showPlot()

    def showPlot(self):
        event = self.el.loadEvent(self.eventList[self.current_i])
        vis_3d = (Clue3DVisualization(event)
            .add3DClusters()
            .add2DClusters()
            .addRechits(hiddenByDefault=True)
            .addImpactTrajectory()
            .addDetectorCylinder()
            .addSliders()
        )
        clear_output(wait=False)
        display(self.button_container)
        pio.show(vis_3d.fig, renderer="notebook_connected")
    

class SampleChooser:
    def __init__(self, df:pd.DataFrame, nSamples=10):
        yes = widgets.Button(description="Yes")
        no = widgets.Button(description="No")
        skip = widgets.Button(description="Skip")
        self.button_container = widgets.HBox([yes, no, skip])

        samples = df.sample(n=nSamples)
        self.el = EventLoader('/data_cms_upgrade/cuisset/testbeam18/clue3d/v31/cmssw/data/CLUE_clusters.root')
        self.itertuples = samples.itertuples()
        self.selectionResults = {}

        #self.current_output = widgets.Output()
        #self.next_output = widgets.Output()
        #self.outputWidget = widgets.Output()
        #display(self.button_container, self.outputWidget)
        self.renderer = NotebookRenderer()
        self.renderer.activate()

        self.preloadNextEvent()
        yes.on_click(self.yesPressed)
        no.on_click(self.noPressed)
        skip.on_click(self.skipPressed)
        
    
    def displayButtons(self):
        display(self.button_container)
    
    def mainCell(self):
        self.mainDisplay = display(display_id="main-display")
        self.next()

    def yesPressed(self, button):
        self.selectionResults[self.current_eventID] = True
        self.next()
    def noPressed(self, button):
        self.selectionResults[self.current_eventID] = False
        self.next()
    def skipPressed(self, button):
        self.selectionResults[self.current_eventID] = None
        self.next()

    def preloadNextEvent(self):
        try:
            #self.next_output.clear_output()
            
            row = next(self.itertuples)
            self.next_eventID = EventID(row.ntupleNumber, row.event)
            event = self.el.loadEvent(self.next_eventID)
            self.next_vis_3d = (Clue3DVisualization(event)
                .add3DClusters()
                .add2DClusters()
                .addRechits(hiddenByDefault=True)
                .addImpactTrajectory()
                .addDetectorCylinder()
                .addSliders()
            )
            self.next_fig_dict = self.next_vis_3d.fig.to_dict()
            #self.next_output = PlotlyRenderer().to_mimebundle(self.next_vis_3d.fig.to_dict())
            #self.next_output.append_display_data(mime_output)
        except StopIteration:
            pass
    
    def next(self):
        self.current_eventID = self.next_eventID
        #print("next")

        #clear_output(wait=False)
        #self.current_output, self.next_output = self.next_output, self.current_output
        #self.current_output = self.next_output
        
        # with self.outputWidget:
        #     clear_output(wait=True)
        #     #display(self.current_output, raw=True)
        #     pio.show(self.next_fig_dict)
        #display(self.button_container, self.current_output, clear=True)
        
        #pio.show(self.next_fig_dict, validate=False)
        self.mainDisplay.update(self.renderer.to_mimebundle(self.next_fig_dict), raw=True)
        #self.renderer.activate()
        self.preloadNextEvent()
        #print("loaded next")

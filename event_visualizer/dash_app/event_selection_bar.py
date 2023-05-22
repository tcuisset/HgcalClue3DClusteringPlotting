import traceback
import sys
from timeit import default_timer as timer

import dash
from dash import Dash, html, Input, Output, State, dcc

from event_visualizer.event_index import EventLoader

legendDivStyle = {'flex': '0 1 auto', 'margin':"10px"}
dropdownStyle = {'flex': '1 1 auto'}


def makeEventSelectionBarComponent(clueParamsList, beamEnergies):
    """ Makes a html Div holding all the dropdowns to select an event """
    return html.Div(children=[
            html.Div("ClueParams :", style=legendDivStyle),
            dcc.Dropdown(options=clueParamsList, id="clueParam", style=dropdownStyle, value="cmssw"),
            dcc.Dropdown(id="datatype", style={"width":"200px"}, value="data"),
            html.Div("Beam energy (GeV) :", style=legendDivStyle),
            dcc.Dropdown(options=beamEnergies, value=None, id="beamEnergy", style=dropdownStyle),
            html.Div("Ntuple number :", style=legendDivStyle),
            dcc.Loading(dcc.Dropdown(id="ntupleNumber"), parent_style=dropdownStyle),
            html.Div("Event :", style=legendDivStyle),
            dcc.Loading(dcc.Dropdown(id="event"), parent_style=dropdownStyle),
            dcc.Clipboard(id="link-copy-clipboard", title="Copy link", content="abc", style={"margin":"5px 10px 2px 10px"}),
        ], 
        style={"display":"flex", "flexFlow":"row"}
    )

def registerEventSelectionBarCallbacks(availableSamples:dict[str, dict[str, EventLoader]]):
    """ Register all the callbacks internal to the event selection bar, that update the dropdown options (datatypes, ntuples, event numbers) 
    The url / clipboard is not handled here
    Parameters : 
     - availableSamples : nested dict clueParam -> datatype -> EventLoader object
    """
    @dash.callback(
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
            print("Failed updating datatypes", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return []

    @dash.callback(
        Output("ntupleNumber", "options"),
        [Input("clueParam", "value"), Input("datatype", "value"), Input("beamEnergy", "value")]
    )
    def update_ntupleNumber(clueParam, datatype, beamEnergy):
        print("update_ntupleNumber", dash.ctx.triggered_prop_ids, dash.ctx.inputs, flush=True)
        start_time = timer()
        try:
            ntuple_energy_pairs = availableSamples[clueParam][datatype].ntuplesEnergies
            dash.callback_context.record_timing('computingNtuplesMap', timer() - start_time, 'Computing list of energy-ntuple pairs')
            return list(ntuple_energy_pairs.ntupleNumber[ntuple_energy_pairs.beamEnergy == beamEnergy])
        except:
            print("Failed updating ntupleNumber", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return []

    @dash.callback(
        Output("event", "options"),
        [Input("clueParam", "value"), Input("datatype", "value"), Input("beamEnergy", "value"), Input("ntupleNumber", "value")],
    )
    def update_availableEvents(clueParam, datatype, beamEnergy, ntupleNumber):
        print("update_availableEvents", dash.ctx.triggered_prop_ids, dash.ctx.inputs, flush=True)
        if None in [clueParam, datatype, beamEnergy, ntupleNumber]:
            return []
        try:
            return list(availableSamples[clueParam][datatype].eventNumbersPerNtuple(beamEnergy, ntupleNumber))
        except:
            print("Failed updated event", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return []


    #When the event dropdown options change, clear the event value. In turn, this will update the event display. 
    #It is needed as otherwise, just changing dropdown options does not cause Input(.., "value") callbacks to be fired
    #leading to display not being synced to event bar
    # We need to check if the current event value is valid, as otherwise on initial call from URL the event value gets overwritten
    dash.clientside_callback(
        """ 
        function(eventOptions, eventValue) {
            if (eventOptions.includes(eventValue)) {
                throw window.dash_clientside.PreventUpdate;
                //return window.dash_clientside.no_update; //does not work for some reason
            } else {
                return null;
            }
        }
        """,
        Output("event", "value", allow_duplicate=True),
        [Input("event", "options"), State("event", "value")],
        prevent_initial_call=True
    )
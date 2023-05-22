import io
import base64

from dash import Dash, html, Input, Output, dcc
import matplotlib.pyplot as plt

import sys

import numpy as np
import matplotlib
matplotlib.use("agg") # no GUI
import matplotlib.pyplot as plt
import matplotlib.ticker
import mplhep as hep
plt.style.use(hep.style.CMS)
import hist
import hist.plot

sys.path.append("../..")
from HistogramLib.histogram import HistogramKind
from HistogramLib.store import HistogramStore
from hists.parameters import beamEnergies
from hists.store import HistogramId

app = Dash(__name__)

app.layout = html.Div([
    html.H1(children='CLUE3D transverse profile'),
    #dcc.Dropdown(options=beamEnergies, value=100, id='beamEnergy', clearable=False),
    dcc.Slider(min=beamEnergies[0], max=beamEnergies[-1], step=None, 
        marks={beamEnergy:f"{beamEnergy} GeV" for beamEnergy in beamEnergies},
        id="beamEnergy", value=100),
    dcc.Slider(min=1, max=28, step=1, value=5, id="layer"),
    html.Div(children=[
        dcc.RadioItems(options={"default" : "Default", "ratio" : "Ratio"}, value="default", inline=True, id="ratioPlot"),
        #dcc.RadioItems(options=[""])
        dcc.Slider(min=1, max=10, step=10/30, value=6, id="maxDistanceToPlot"),
    ]),
    
    html.Img(id="plot")
])


hist_folder = '/grid_mnt/data_cms_upgrade/cuisset/testbeam18/clue3d/v33'
#clueParams = "single-file"
clueParams = "cmssw"
histStore = HistogramStore(hist_folder, HistogramId)
datatypeToLegendMap = {"data":"Data", "sim_proton_v46_patchMIP":"Simulation"}


def loadHists(layer:int, beamEnergy:int, datatypes:list[str]=["data", "sim_proton_v46_patchMIP"]) -> tuple[list[hist.Hist], list[str]]:
    return [(histStore
        .get(HistogramId("Clus3DRechitsDistanceToImpact_AreaNormalized", clueParams, datatype))
        .getHistogram(HistogramKind.WEIGHTED_PROFILE)[{
            "beamEnergy" : hist.loc(beamEnergy),
            "mainOrAllTracksters" : hist.loc("mainTrackster"),
            # Project on clus3D_size
            "layer" : hist.loc(layer),
        }]
        .project("rechits_distanceToImpact")
        ) for datatype in datatypes], [datatypeToLegendMap.get(datatype, datatype) for datatype in datatypes]

def addLumiLegend(main_ax, datatypes, layer, beamEnergy):
    if "data" in datatypes:
        hep.cms.text("Preliminary", ax=main_ax)
    else:
        hep.cms.text("Simulation Preliminary", ax=main_ax)
    hep.cms.lumitext(f"Layer {layer} - $e^+$ {str(beamEnergy)} GeV", ax=main_ax)
    main_ax.legend()

def makePlotMultiDatatype(layer:int, beamEnergy:int, datatypes:list[str], maxDistanceToPlot=6):
    """ Plot distribution of distance to impact on a layer 
    See in custom_hists for how y is computed"""
    hists, labels = loadHists(layer, beamEnergy, datatypes)
    yerr = False
    
    #fig = plt.Figure()
    fig = plt.figure()
    ax = fig.subplots()
    ax.set_xlabel("Distance to extrapolated impact point (cm)")
    ax.set_xlim(0, maxDistanceToPlot)
    ax.set_ylim(3e-4, 3)
    ax.set_ylabel(r"$\frac{1}{E_{cluster}} \frac{dE_{hit}}{dA} (cm^{-2})$")
    ax.set_yscale("log")
    
    hep.histplot(hists, label=labels, yerr=yerr, ax=ax)

    addLumiLegend(ax, datatypes, layer, beamEnergy)
    return fig

def makePlotRatio(layer:int, beamEnergy:int, datatypes:list[str], maxDistanceToPlot=6):
    if len(datatypes) != 2:
        raise RuntimeError()
    hists, labels = loadHists(layer, beamEnergy, datatypes)

    #fig = plt.Figure()
    fig = plt.figure()
    grid = fig.add_gridspec(2, 1, hspace=0, height_ratios=[3, 1])
    main_ax:plt.Axes = fig.add_subplot(grid[0])
    subplot_ax:plt.Axes = fig.add_subplot(grid[1], sharex=main_ax)

    hep.histplot(hists, label=labels, yerr=False, ax=main_ax)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = hists[0].values() / hists[1].values()
        #ratio_uncert = hist.intervals.ratio_uncertainty(
        #    num=hists[0].values(),
        #    denom=hists[1].values(),
        #    uncertainty_type="poisson", # Assume numerator is Poisson (ignore uncertainty on MC)
        #)
        ratio_uncert = None
        hist.plot.plot_ratio_array(hists[0], ratios, ratio_uncert, subplot_ax, ylim=(0.2,2), ylabel="Ratio")
    
    #plt.gca().xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    #main_ax.xaxis.set_major_locator(matplotlib.ticker.AutoLocator())
    #main_ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.setp(main_ax.get_xticklabels(), visible=False)
    main_ax.set_ylabel(r"$\frac{1}{E_{cluster}} \frac{dE_{hit}}{dA} (cm^{-2})$")
    main_ax.set_yscale("log")
    main_ax.set_xlabel("")
    main_ax.set_xlim(0, maxDistanceToPlot)
    subplot_ax.set_xlabel("Distance to extrapolated impact point (cm)")

    addLumiLegend(main_ax, datatypes, layer, beamEnergy)
    return fig

def mplFigureToUrl(fig=None):
    if fig is None:
        fig = plt.gcf()
    buf = io.BytesIO() # in-memory files
    fig.savefig(buf, format="png")
    plt.close(fig)
    data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    buf.close()
    return "data:image/png;base64,{}".format(data)


@app.callback(
    Output(component_id='plot', component_property='src'),
    [Input(component_id = 'beamEnergy', component_property='value'),
    Input("layer", "value"), Input("ratioPlot", "value"), Input("maxDistanceToPlot", "value")]
)
def update_graph(beamEnergy, layer, ratioPlot:bool, maxDistanceToPlot):
    if ratioPlot == "ratio":
        fig = makePlotRatio(layer, beamEnergy=beamEnergy, datatypes=["data", "sim_proton_v46_patchMIP"], maxDistanceToPlot=maxDistanceToPlot)
    else:
        fig = makePlotMultiDatatype(layer, beamEnergy=beamEnergy, datatypes=["data", "sim_proton_v46_patchMIP"], maxDistanceToPlot=maxDistanceToPlot)
    return mplFigureToUrl(fig)

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
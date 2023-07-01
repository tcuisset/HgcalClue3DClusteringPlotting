
import numpy as np
import hist
import matplotlib.ticker
import matplotlib.pyplot as plt
import mplhep as hep

def _convert2DHist(h_2D:hist.Hist|dict[int, hist.Hist]) -> tuple[list[int], list[hist.Hist]]:
    if isinstance(h_2D, hist.Hist):
        return list(h_2D.axes["beamEnergy"]), [h_2D[{"beamEnergy" : hist.loc(beamEnergy)}] for beamEnergy in h_2D.axes["beamEnergy"]]
    else:
        return list(h_2D.keys()), h_2D.values()


def makeViolinBeamEnergy(h_2D:hist.Hist|dict[int, hist.Hist], datatype=None, ax=None, lumitext="$e^+$ Test Beam"):
    """ Plots for each beam energy, and each layer, the mean (over all events) of the 3D clustered energy in the layer
    Parameters : 
     - h_2D : can be : either an histogram with axes layer, beamEnergy and with value the mean energy on the layer
           - either a dict beamEnergy -> 1D histogram with axis layer and with value the mean energy on layer
     - datatype : can be "data", None, or anything else (-> simulation)
    """
    beamEnergies, hists  = _convert2DHist(h_2D)

    if ax is None:
        fig, ax = plt.subplots()
    max_height = max((np.max(h.density()) for h in hists))*1.2
    y_locations = np.arange(0, max_height*len(hists), max_height)
    for x_loc, h in zip(y_locations, hists):
        ax.bar(x=[h.axes[0].bin(i) for i in range(h.axes[0].size)], height=h.density(), bottom=(x_loc - 0.5 * h.density()), width=1.)
    
    ax.set_xlabel("Layer")
    ax.set_ylabel("Nominal beam energy (GeV)")
    ax.set_ylim(bottom=y_locations[0]-max_height/2*1.3)
    ax.set_yticks(y_locations, beamEnergies)
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))

    if "data" == datatype:
        hep.cms.text("Preliminary", ax=ax)
    elif datatype is not None:
        hep.cms.text("Simulation Preliminary", ax=ax)
    if lumitext is not None:
        hep.cms.lumitext(lumitext, ax=ax)

def makeViolinBeamEnergyDataSim(h_2D_data:hist.Hist|dict[int, hist.Hist], h_2D_sim:hist.Hist|dict[int, hist.Hist], ax=None, lumitext="$e^+$ Test Beam"):
    """ Same as makeViolinBeamEnergy but overlays simulation on top as a dashed line """
    beamEnergies, hists = _convert2DHist(h_2D_data)
    beamEnergies_sim, hists_sim = _convert2DHist(h_2D_sim)
    assert beamEnergies == beamEnergies_sim

    if ax is None:
        fig, ax = plt.subplots()
    max_height = max((np.max(h.density()) for h in hists))*1.2
    y_locations = np.arange(0, max_height*len(hists), max_height)
    for x_loc, h, h_sim in zip(y_locations, hists, hists_sim):
        ax.bar(x=[h.axes[0].bin(i) for i in range(h.axes[0].size)], height=h.density(), bottom=(x_loc - 0.5 * h.density()), 
               width=1.,  label="Data" if x_loc == y_locations[-1] else None)
        step_kwargs = dict(x=[h.axes[0].bin(i) for i in range(h.axes[0].size)], where="mid", 
        color="black", linestyle="--", alpha=0.5)
        ax.step(y=(x_loc + 0.5 * h_sim.density()),  **step_kwargs, label="Simulation" if x_loc == y_locations[0] else None)
        ax.step(y=(x_loc - 0.5 * h_sim.density()),  **step_kwargs)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Nominal beam energy (GeV)")
    ax.set_ylim(bottom=y_locations[0]-max_height/2*1.3)
    ax.set_yticks(y_locations, beamEnergies)
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))

    hep.cms.text("Preliminary", ax=ax)
    if lumitext is not None:
        hep.cms.lumitext(lumitext, ax=ax)
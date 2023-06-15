import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
import numpy as np



def scatterPredictionVsTruth(true_values:np.ndarray, pred_values:np.ndarray, epoch=None) -> mpl.figure.Figure:
    fig, ax = plt.subplots(figsize=(20, 20), dpi=50)
    ax.scatter(true_values, pred_values)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes) # oblique line

    ax.set_xlim(left=0)
    ax.set_xlabel("True beam energy (GeV)")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Predicted beam energy (GeV)")

    hep.cms.text("Preliminary Simulation", ax=ax)
    if epoch is not None:
        hep.cms.lumitext(f"Epoch : {epoch}", ax=ax)
    return fig

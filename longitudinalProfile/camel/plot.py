import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)


def plotIndividualProfile(energyPerLayer_df:pd.DataFrame, eventNb, maximas_df:pd.DataFrame=None):
    fig, ax = plt.subplots()
    ax.plot(list(range(1, 28+1)), energyPerLayer_df.rechits_energy_sum_perLayer.loc[eventNb], '+-', markersize=6)
    if maximas_df is not None:
        ax.scatter(x=[maximas_df.maximaLayer_0.loc[eventNb], maximas_df.maximaLayer_1.loc[eventNb]], 
                   y=[maximas_df.rechits_energy_sum_perLayer_0.loc[eventNb], maximas_df.rechits_energy_sum_perLayer_1.loc[eventNb]],
                   color="green")
        ax.scatter(x=[maximas_df.dip_layer.loc[eventNb]], y=[maximas_df.dip_layer_energy.loc[eventNb]],
                   color="red")

    ax.set_xlabel("Layer number")
    ax.set_ylabel("Energy on layer (GeV)")
    ax.tick_params(axis="x", labeltop=True)


def plotIndividualProfile_scipy(energyPerLayer_series:pd.Series, peaks, properties):
    fig, ax = plt.subplots()
    ax.plot(energyPerLayer_series.index, energyPerLayer_series, '+-', markersize=6, label="Energy on layer (GeV)")
    ax.scatter(x=peaks+1, y=energyPerLayer_series.iloc[peaks], color="green", label="Maximas")

    if properties["width_heights"][0] < 0:
        reverse = True
        factor = -1
    else:
        reverse = False
        factor = 1

    ax.hlines(factor * properties["width_heights"], properties["left_ips"]+1, properties["right_ips"]+1, 
              colors=["purple", "magenta"], label="width_heights")
    
    ax.vlines(x=peaks+1, ymin=energyPerLayer_series.iloc[peaks] - factor*properties["prominences"], ymax=energyPerLayer_series.iloc[peaks],
              color="orange", label="Prominence height")
    ax.scatter(x=properties["left_bases"]+1, y=energyPerLayer_series.iloc[properties["left_bases"]], marker="<", color="purple", label="Left bases")
    ax.scatter(x=properties["right_bases"]+1, y=energyPerLayer_series.iloc[properties["right_bases"]], marker=">", color="purple", label="Right bases")
    ax.legend()
    ax.set_xlabel("Layer number")
    ax.set_ylabel("Energy on layer (GeV)")
    #ax.plot([], [], " ", label=f"")
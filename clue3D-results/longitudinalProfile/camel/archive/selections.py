import functools

import numpy as np
import pandas as pd
import scipy
import uproot

from hists.dataframe import DataframeComputations


def maxima_indices(s:pd.Series, order=3):
    return pd.Series(scipy.signal.argrelmax(s.to_numpy(), order=order, mode='clip')[0]+1) # +1 to map layer from indexing 0-based to 1-based 
def computeMaximasDf(energyPerLayer_series:pd.Series, order=3):
    return (energyPerLayer_series
            .groupby(by="eventInternal").apply(functools.partial(maxima_indices, order=order))

            .groupby(by="eventInternal").filter(lambda x:len(x)>1)
            .rename_axis(["eventInternal", "maximaIndex"])
            .rename("maximaLayer")
    )

def cutOnFractionOfMaxLayerEnergy(maximas_series:pd.Series, energySumPerLayer_df:pd.DataFrame, fractionOfMaxLayerEnergy):
    """ Filter all maximas that are on layer that have less than a fraction than the max layer energy """
    # Join with energySumPerLayer_df so we get the energy sum on layer for each maxima layer
    merged_df = (pd.merge(maximas_series, energySumPerLayer_df,
         left_on=["eventInternal", "maximaLayer"], right_index=True,
         how="left")
    )
    # Add column rechits_energy_sum_perLayer_maxPerEvent which is the energy of the layer which has max energy per event
    merged_df = merged_df.join(merged_df.rechits_energy_sum_perLayer.groupby("eventInternal").max().rename("rechits_energy_sum_perLayer_maxPerEvent"))
    
    return (
        # Filter all maximas that are on layer that have less than a fraction than the max layer energy
        merged_df[merged_df.rechits_energy_sum_perLayer >= fractionOfMaxLayerEnergy * merged_df.rechits_energy_sum_perLayer_maxPerEvent]
        # Remove events which have only one maxima after this cut
        .groupby(by="eventInternal").filter(lambda x:len(x)>1)
    )

def addInfo(cut_df:pd.DataFrame, comp:DataframeComputations):
    return cut_df.join(comp.ntupleEvent)

def filterLowEnergyEvents(df:pd.DataFrame, fractionOfBeamEnergy=0.6):
    return df[df.rechits_energy_sum >= fractionOfBeamEnergy*df.beamEnergy]

def selectTwoHighestMaxima(df:pd.DataFrame):
    return df.sort_values(["eventInternal", "rechits_energy_sum_perLayer"], ascending=[True, False]).groupby("eventInternal").head(2)

def unstackTwoLevels(df:pd.DataFrame):
    """ Unstack the two maximas into columns """
    #unstacked =  df.unstack()
    allColumns = set(df.columns)
    singleColumns = set(("rechits_energy_sum", "rechits_energy_sum_perLayer_maxPerEvent",
         "ntupleNumber", "event", "beamEnergy")).intersection(allColumns)
    
    # Single columns : take maximaIndex = 0
    out_dict = {col : df[col].loc[:, 0] for col in singleColumns}
    
    for doubleCol in allColumns.difference(singleColumns):
        for i in range(2):
            out_dict[doubleCol + "_" + str(i)] = df[doubleCol].loc[:, i]
    
    return pd.DataFrame(out_dict)



def computeDipDepthAndEnergyQuantile(df:pd.DataFrame, energySumPerLayer_df:pd.DataFrame, quantile=0.5):
    # First join with all erngy sums on all layers
    dip_df = df[["maximaLayer_0", "maximaLayer_1"]].join(energySumPerLayer_df).reset_index("rechits_layer")
    # Filter to keep only energy sums in between maximas
    dip_df = dip_df[(dip_df.rechits_layer > dip_df.maximaLayer_0) & (dip_df.rechits_layer < dip_df.maximaLayer_1)]


    # Keep only lowest value of rechits_energy_sum_perLayer
    lowestValueInDip_df = dip_df.sort_values(["eventInternal", "rechits_energy_sum_perLayer"], ascending=[True, True])
    lowestValueInDip_df = lowestValueInDip_df[~lowestValueInDip_df.index.duplicated(keep="first")].rename(columns={"rechits_energy_sum_perLayer":"dip_layer_energy", "rechits_layer":"dip_layer"})
    
    quantile_series = dip_df.set_index("rechits_layer", append=True).rechits_energy_sum_perLayer.groupby("eventInternal").quantile(q=quantile)
    lowestValueInDip_df["energyQuantileInDip"] = quantile_series
    new_df = df.join(lowestValueInDip_df[["dip_layer_energy", "dip_layer", "energyQuantileInDip"]])

    new_df["meanOfMaximasEnergies"] = new_df[["rechits_energy_sum_perLayer_0", "rechits_energy_sum_perLayer_1"]].mean(axis="columns")
    return new_df

def filterDipDepth(df:pd.DataFrame, fraction):
    """ Filter events such that the min energy in the dip has to be at less than fraction * energy at both maximas """
    return df[(df.dip_layer_energy <= fraction * df.rechits_energy_sum_perLayer_0)&(df.dip_layer_energy <= fraction * df.rechits_energy_sum_perLayer_1)]


def computeDipLength(df:pd.DataFrame):
    """ Only works for 2 maximas """
    return df.assign(dipLength=df.maximaLayer_1-df.maximaLayer_0)

def filterDipLength(df:pd.DataFrame, minDipLength):
    return df[df.dipLength >= minDipLength]


def filterDipDepthComparedToMaximas(df:pd.DataFrame):
    """ Filter events so that the difference in energy between the maximas has to be smaller than the difference between the lowest maxima and the dip energy 
    """
    abs_diff_maximas = np.abs(df.rechits_energy_sum_perLayer_0 - df.rechits_energy_sum_perLayer_1)
    lowest_maxima = df[["rechits_energy_sum_perLayer_0", "rechits_energy_sum_perLayer_1"]].min(axis="columns")
    return df[abs_diff_maximas < lowest_maxima - df.dip_layer_energy]

def filterDipQuartile(df:pd.DataFrame, fraction=0.6):
    """ Filter events so that the quantile of energy in between the maxima is less than fraction of the highest maxima """
    return df[df.energyQuantileInDip <= fraction * df.rechits_energy_sum_perLayer_maxPerEvent]
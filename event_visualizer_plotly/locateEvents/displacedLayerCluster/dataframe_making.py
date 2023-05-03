import math
from functools import partial
import uproot
import numpy as np
import pandas as pd
import hist
from tqdm import tqdm

from hists.parameters import thresholdW0 as defaultThresholdW0
from hists.parameters import synchrotronBeamEnergiesMap
from hists.dataframe import DataframeComputations
from hists.store import HistogramId
from hists.custom_hists import beamEnergiesAxis, layerAxis
from HistogramLib.store import HistogramStore


########## Building dataframe

clus2D_columns = ["beamEnergy", "clus2D_x", "clus2D_y", "clus2D_layer", "clus2D_energy", "clus2D_size"]

def makeBarycenterDf(comp:DataframeComputations, thresholdW0=defaultThresholdW0):
    """ Make barycenter of rechits dataframe """
    return comp.computeBarycenter(
        (comp.clusters2D_merged_rechits(frozenset(["rechits_x", "rechits_y", "rechits_energy", "rechits_layer"]), useCachedRechits=False,
            clusters2D_columns=[])
         .set_index(["rechits_id"], append=True)), 
        groupbyColumns=["eventInternal","clus2D_id"], thresholdW0=thresholdW0
    )

def barycenterDifferenceFromDefault(comp:DataframeComputations):
    """ Compute distance between CLUE LC position and barycenter position """
    df_barycenter_noDistance = makeBarycenterDf(comp)
    return (comp.clusters2D_custom(clus2D_columns)
        .assign(clus2D_x_barycenter=df_barycenter_noDistance.rechits_x_barycenter,
            clus2D_y_barycenter=df_barycenter_noDistance.rechits_y_barycenter)
        .eval(
            "clus2D_distance_positionToBarycenter = "
            " sqrt((clus2D_x - clus2D_x_barycenter)**2 + (clus2D_y - clus2D_y_barycenter)**2)"
        )
        # Add rechits_energy_sum_perLayer column
        .join(comp.rechits_totalReconstructedEnergyPerEventLayer, on=["eventInternal", "clus2D_layer"], how="left")
    )

def joinWithImpact(barycenterDistance_df:pd.DataFrame, comp:DataframeComputations):
    return (barycenterDistance_df
        .join(comp.impact, on=["eventInternal", "clus2D_layer"])
        .eval(
            "clus2D_distance_positionToImpact = "
            " sqrt((clus2D_x - impactX)**2 + (clus2D_y - impactY)**2)"
            "\n"
            "clus2D_distance_barycenterToImpact = "
            " sqrt((clus2D_x_barycenter - impactX)**2 + (clus2D_y_barycenter - impactY)**2)"
        )
    )

def joinWithClusters3D(df:pd.DataFrame, comp:DataframeComputations):
    return pd.merge(
        df, 
        comp.clusters3D_with_clus2D_id().reset_index("clus3D_id"), 
        how="left",
        left_index=True, right_on=["eventInternal", "clus2D_id"]
    )

def makeFullBarycenterDf(comp:DataframeComputations):
    return barycenterDifferenceFromDefault(comp).pipe(joinWithImpact, comp=comp).pipe(joinWithClusters3D, comp=comp)


############## Main algo

def findDisplacedLC(tree:uproot.TTree, lookupFunctions:list, maxCount=math.inf):
    """ Find displaced LC
    Parameters : 
     - lookupFunctions : a list of functions, taking as argument : 
            barycenterDistance_df (output of barycenterDifferenceFromDefault), comp (current DataframeComputations)
     - maxCount : stop early after finding this event count
    """
    filtered_df = {fct.__name__ : [] for fct in lookupFunctions}
    counts = {fct.__name__ : 0 for fct in lookupFunctions}
    with tqdm(total=tree.num_entries) as pbar:
        for array, report in tree.iterate(step_size="50MB", library="ak", report=True,
                filter_name=["event", "ntupleNumber", "beamEnergy", "rechits_x", "rechits_y", "rechits_layer", "rechits_energy", "clus2D_idxs"]
                    +clus2D_columns+["impactX", "impactY", "clus3D_energy", "clus3D_size", "clus3D_idxs"]):
            comp = DataframeComputations(array, ["rechits_x", "rechits_y", "rechits_layer", "rechits_energy"])
            barycenterDistance_df = makeFullBarycenterDf(comp)

            for lookupFct in lookupFunctions:
                out_df = lookupFct(barycenterDistance_df, comp).join(comp.ntupleEvent, rsuffix="_r")
                filtered_df[lookupFct.__name__].append(out_df)
                counts[lookupFct.__name__] += out_df.size
            
            if max(counts.values()) >= maxCount:
                break

            pbar.update(report.stop-report.start)

    return {fctName : pd.concat(dfList) for fctName, dfList in filtered_df.items()}


def fillHistogram(df:pd.DataFrame):
    h = hist.Hist(beamEnergiesAxis(), layerAxis, storage=hist.storage.Int64())
    h.fill(df.beamEnergy, df.clus2D_layer)
    return h
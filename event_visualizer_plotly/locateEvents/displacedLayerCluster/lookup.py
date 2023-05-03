import pandas as pd

from hists.dataframe import DataframeComputations

########### Lookup functions

def lookupLargeDistancePositionBarycenter(df:pd.DataFrame, _:DataframeComputations, minDistance=3):
    """ Select from dataframe LC where distance from CLUE position to barycenter is above minDistance (cm) """
    return (df
        .query("clus2D_distance_positionToBarycenter >= @minDistance")
    )


def lookupLargeDistanceToImpactNotInBarycenter(fullBarycenterDistance_df:pd.DataFrame, comp:DataframeComputations,
        minDistancePositionToBarycenter=2, maxDistanceBarycenterToImpact=1, minDistancePositionToImpact=1, beamEnergyFraction=0.2/28):
    """ Find LC where the barycenter gives a position close to DWC impact but CLUE LC position is off """
    return (
        fullBarycenterDistance_df
        .query("clus2D_distance_positionToBarycenter >= @minDistancePositionToBarycenter"
               " and clus2D_distance_barycenterToImpact < @maxDistanceBarycenterToImpact"
               " and clus2D_distance_positionToImpact > @minDistancePositionToImpact"
               " and clus2D_energy >= @beamEnergyFraction * beamEnergy")
    )

def lookupFractionEnergyOnLayer(df:pd.DataFrame, comp:DataframeComputations,
        minDistancePositionToBarycenter=2, maxDistanceBarycenterToImpact=1, minDistancePositionToImpact=1, 
        minLayerEnergyFraction=0.7, max3DClusterSize=20):
    """ Find LC where the 
     - barycenter gives a position close to DWC impact but CLUE LC position is off
     - the LC has at least minLayerEnergyFraction fraction of energy of total rechits on layer (default 70% min)
     - the LC has either no associated 3D cluster or a 3D cluster with less than max3DClusterSize (default 20) LC
    """
    df = df.query(
        "clus2D_distance_positionToBarycenter >= @minDistancePositionToBarycenter"
        " and clus2D_distance_barycenterToImpact < @maxDistanceBarycenterToImpact"
        " and clus2D_distance_positionToImpact > @minDistancePositionToImpact"
        #" and clus2D_energy >= @beamEnergyFraction * beamEnergy"
        " and clus2D_energy >= @minLayerEnergyFraction * rechits_energy_sum_perLayer"
    )
    # Select LC with either no associated 3D cluster (outliers or follower of outlier or masked cluster),
    # either in a 3D cluster with less than max3DClusterSize LC
    df = df[(df.clus3D_size.isna()) | (df.clus3D_size <= max3DClusterSize)]
    return df
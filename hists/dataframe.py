""" 
Helper class for reading CLUE_clusters.root file 
eventInternal column is not the same as "event" branch in ntuples, it is an ID that starts from 0 in each batch of entries
"""
import math
from functools import cached_property, partial
import operator
from typing import Any, Iterable

import numpy as np
import pandas as pd
import awkward as ak

from . import parameters
from .parameters import synchrotronBeamEnergiesMap, layerToZMapping
from .shower_axis import computeAngleSeries

import functools
import weakref

def memoized_method(*lru_args, **lru_kwargs):
    """Small code taken from https://stackoverflow.com/a/33672499 
    So we can cache method results with cache that is per instance rather than per class
    as using @functools.cache leads to memery leaks when used on an instance method"""
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(self, *args, **kwargs):
            # We're storing the wrapped method inside the instance. If we had
            # a strong reference to self the instance would never die.
            self_weak = weakref.ref(self)
            @functools.wraps(func)
            @functools.lru_cache(*lru_args, **lru_kwargs)
            def cached_method(*args, **kwargs):
                return func(self_weak(), *args, **kwargs)
            setattr(self, func.__name__, cached_method)
            return cached_method(*args, **kwargs)
        return wrapped_func
    return decorator


def divideByBeamEnergy(df:pd.DataFrame, colName:str) -> pd.DataFrame:
    """ Add a columns named colName_fractionOfSynchrotronBeamEnergy """
    df[colName+"_fractionOfSynchrotronBeamEnergy"] = df[colName]/df["synchrotronBeamEnergy"]
    return df


class DataframeComputations:
    def __init__(self, tree_array:ak.Array, rechits_columns=["beamEnergy", "rechits_x", "rechits_y", "rechits_z", "rechits_energy", "rechits_layer",
            "rechits_rho", "rechits_delta", "rechits_nearestHigher", "rechits_pointType"]) -> None:
        """ Parameters : 
         - tree_array : an awkward Array of events data
         - rechits_columns : columns loaded when calling rechits 
        """
        self.array = tree_array
        self.rechits_columns = rechits_columns
    
    @cached_property
    def ntupleEvent(self) -> pd.DataFrame:
        return ak.to_dataframe(self.array[["ntupleNumber", "event", "beamEnergy"]], 
            levelname=lambda i : {0 : "eventInternal"}[i])

    @cached_property
    def trueBeamEnergy(self) -> pd.DataFrame:
        """ Get simulated particle gun energy branch. If trueBeamEnergy branch exists, returns : 
            Columns : eventInternal beamEnergy, trueBeamEnergy
            Index : eventInternal
        otherwise return a dataframe with a single row : 1, 0, 0
        """
        if "trueBeamEnergy" in self.array.fields:
            return (ak.to_dataframe(self.array[
                ["beamEnergy", "trueBeamEnergy"]
                ], 
                levelname=lambda i : {0 : "eventInternal"}[i])
            )
        else:
            return pd.DataFrame({"eventInternal":[1], "beamEnergy":[0], "trueBeamEnergy":[0]}).set_index("eventInternal")

    @cached_property
    def beamEnergy(self) -> pd.DataFrame:
        """ 
        Index : eventInternal
        Columns : beamEnergy synchrotronBeamEnergy (ie mean particle energy in beam test, after synchrotron losses)
        """
        df =  ak.to_dataframe(
            self.array.beamEnergy, 
            levelname=lambda i: "eventInternal", 
            anonymous="beamEnergy" # name the column
        )
        df["synchrotronBeamEnergy"] = df.beamEnergy.map(synchrotronBeamEnergiesMap)
        return df
    

    def join_divideByBeamEnergy(self, df:pd.DataFrame, colName:str) -> pd.DataFrame:
        """ Joins to beamEnergy df then adds a columns named colName_fractionOfSynchrotronBeamEnergy"""
        # rsuffix is in case we already have beamEnergy column in df
        return df.join(self.beamEnergy, on="eventInternal", rsuffix="_right").pipe(divideByBeamEnergy, colName=colName)

    @cached_property
    def impact(self) -> pd.DataFrame:
        """ 
        Columns : eventInternal   layer   impactX  impactY  
        MultiIndex : (eventInternal, layer)
        """
        df = ak.to_dataframe(self.array[
            ["impactX", "impactY"]],
            levelname=lambda i : {0 : "eventInternal", 1:"layer_minus_one"}[i]).reset_index(level="layer_minus_one")
        df["layer"] = df["layer_minus_one"] + 1
        return df.drop("layer_minus_one", axis="columns").set_index("layer", append=True)
    
    @property
    def impactWithBeamEnergy(self) -> pd.DataFrame:
        """ 
        Columns : eventInternal   layer beamEnergy  impactX  impactY  
        Index : eventInternal
        """
        df = ak.to_dataframe(self.array[
            ["beamEnergy", "impactX", "impactY"]],
            levelname=lambda i : {0 : "eventInternal", 1:"layer_minus_one"}[i]).reset_index(level="layer_minus_one")
        df["layer"] = df["layer_minus_one"] + 1
        return df.drop("layer_minus_one", axis="columns")

    def rechits_custom(self, columns:list[str]) -> pd.DataFrame:
        """ Load rechits """
        return ak.to_dataframe(self.array[columns], 
            levelname=lambda i : {0 : "eventInternal", 1:"rechits_id"}[i])

    @cached_property
    def rechits(self) -> pd.DataFrame:
        """
        Columns : eventInternal  rechits_id  beamEnergy	rechits_x	rechits_y	rechits_z	rechits_energy	rechits_layer	rechits_rho	rechits_delta rechits_nearestHigher rechits_pointType  
        MultiIndex : (eventInternal, rechits_id)
        """
        return self.rechits_custom(self.rechits_columns)

    @property
    def rechits_twoMostEnergeticPerLayer(self) -> pd.DataFrame:
        """ Find for each layer the two most energetic hits 
        Returns a dataframe with : 
        Index : eventInternal, rechits_layer
        Columns : rechits_id[0] and [1], rechits_energy[0] and [1]
        NB : rechits_id[1] is -1 in case there is only one rechit on the layer
        """
        df = (self.rechits[["rechits_layer", "rechits_energy"]]
            .set_index("rechits_layer", append=True).reset_index("rechits_id")

            # Select the first two rechits in energy per layer
            .sort_values(["eventInternal", "rechits_layer", "rechits_energy"], ascending=[True, True, False])
            .groupby(["eventInternal", "rechits_layer"]).head(2)
        )
        # unstack the dataframe
        return (df
            # Make a cumulative count, which is 0 for the most energetic hit, 1 for the second (for each event and layer)
            .assign(cumcount=df.groupby(["eventInternal", "rechits_layer"]).cumcount())
            .set_index("cumcount", append=True)
            # Unstack the dataframe (filling rechits_id with -1 in case of only one hit)
            .unstack(fill_value=-1)
        )
    
    @property
    def rechits_ratioFirstToSecondMostEnergeticHitsPerLayer(self) -> pd.Series:
        df = self.rechits_twoMostEnergeticPerLayer
        ratio = df.rechits_energy[0]/df.rechits_energy[1]
        return ratio[ratio > 0].rename("rechits_ratioFirstToSecondMostEnergeticHitsPerLayer") # drop cases where there is only one rechit on layer (then rechits_energy[1] is -1)

    @cached_property
    def rechits_totalReconstructedEnergyPerEvent(self) -> pd.DataFrame:
        """ Sum of all rechits energy per event
        Index : eventInternal
        Columns : beamEnergy rechits_energy_sum
        """
        return (self.rechits[["rechits_energy"]]
            .groupby(by="eventInternal")
            .agg(
                rechits_energy_sum=pd.NamedAgg(column="rechits_energy", aggfunc="sum"),
            )
            .pipe(self.join_divideByBeamEnergy, "rechits_energy_sum")
        )

    @property
    def rechits_totalReconstructedEnergyPerEventLayer(self) ->pd.DataFrame:
        """ Sum of all rechits energy per event and per layer
        Index : eventInternal, rechits_layer
        Columns : rechits_energy_sum_perLayer
        """
        return (self.rechits[["rechits_layer", "rechits_energy"]]
            .groupby(by=["eventInternal", "rechits_layer"])
            .agg(
                rechits_energy_sum_perLayer=pd.NamedAgg(column="rechits_energy", aggfunc="sum"),
            )
        )
    
    def rechits_totalReconstructedEnergyPerEventLayer_allLayers(self, joinWithBeamEnergy=True) -> pd.DataFrame:
        """ Sum of all rechits energy per event and per layer, but with all layers present (filled with zeroes if necessary)
        To compute profile on energy sums correctly, it is necessary to include rows with zeroes in case a layer does not have any rechits
        about 2% increase in nb of rows at 100 GeV (probably much more at 20 GeV)
        Index : eventInternal
        Columns : beamEnergy rechits_energy_sum_perLayer
        """
        df = self.rechits_totalReconstructedEnergyPerEventLayer
        # We build the cartesian product event * layer
        newIndex = pd.MultiIndex.from_product([df.index.levels[0], df.index.levels[1]])

        return_df = (
            # Reindex the dataframe, this will create new rows as needed, filled with zeros
            # Make sure only columns where 0 makes sense are included (not beamEnergy !)
            df.reindex(newIndex, fill_value=0)
            .reset_index("rechits_layer")
        )
        if joinWithBeamEnergy:
            return return_df.pipe(self.join_divideByBeamEnergy, colName="rechits_energy_sum_perLayer")
        else:
            return return_df

    @property
    def rechits_layerWithMaxEnergy(self):
        """ For each event, find the layer with maximum reconstructed energy
        Index : eventInternal
        Columns : rechits_layer	rechits_energy_sum_perLayer 
        """
        return (self.rechits_totalReconstructedEnergyPerEventLayer
            .sort_values(["eventInternal", "rechits_energy_sum_perLayer"], ascending=[True, False])
            .reset_index()
            .drop_duplicates("eventInternal", keep="first")
            .set_index("eventInternal")
            .pipe(self.join_divideByBeamEnergy, "rechits_energy_sum_perLayer")
        )

    @cached_property
    def layerToZMapping(self) -> dict[int, float]:
        """ Returns a dict layer_nb -> layer z position (there is a key for a layer only if there is a rechit at this layer in at least one loaded event)"""
        return self.rechits[["rechits_z", "rechits_layer"]].groupby("rechits_layer").first()["rechits_z"].to_dict()

    def impactWithZPosition(self, useStaticLayerToZMap=False) -> pd.DataFrame:
        """ Transform impact df to use z layer position instead of layer number 
        Parameter : useStaticLayerToZMap : if True, use layer to z map from hists.parameters. If False, compute it from current dataframe
        MultiIndex : (eventInternal, impactZ)
        Columns : impactX, impactY
        """
        if useStaticLayerToZMap:
            layerToZMapping_used = layerToZMapping # from hists.parameters, static dict
        else:
            layerToZMapping_used = self.layerToZMapping # computed from current dataframe
        
        # Select only values of layer that are in the dictionnary layerToZMapping (otherwise you have a column with layer numbers and z positions at the same time)
        df = self.impact.iloc[self.impact.index.get_level_values("layer") <= max(layerToZMapping_used)].rename(
            index=layerToZMapping_used, level="layer")
        df.index.names = ["eventInternal", "impactZ"] # Rename layer -> impactZ
        return df

    def clusters3D_interpolateImpactAtZ(self, clus3D_df=None, useStaticLayerToZMap=False) -> pd.DataFrame:
        """ Interpolate the DWC impact x and y at the clus3D_z position 
        Note : this is quite slow due to the apply call
        Parameters : 
         - clus3D_df : a dataframe of 3D clusters, must have (eventInternal, clus3D_id) index, and at least clus3D_z as column
        Returns : 
        a dataframe indexed by (eventInternal, clus3D_id) with columns impactX_interpolated, impactY_interpolated
        """
        if clus3D_df is None:
            clus3D_df = self.clusters3D

        def interp(group:pd.DataFrame):
            """ To apply on a group of a single 3D cluster """
            return pd.Series([np.interp(x=[group.clus3D_z.iloc[0]], xp=group.impactZ, fp=group.impactX)[0], 
                            np.interp(x=[group.clus3D_z.iloc[0]], xp=group.impactZ, fp=group.impactY)[0]],
                            index=["impactX_interpolated", "impactY_interpolated"])

        return clus3D_df[["clus3D_z"]].join(self.impactWithZPosition(useStaticLayerToZMap=useStaticLayerToZMap)).reset_index("impactZ").groupby(["eventInternal", "clus3D_id"]).apply(interp)

    def clusters2D_custom(self, columnsList:list[str]) -> pd.DataFrame:
        """ Builds a pandas DataFrame holding 2D cluster info (without rechits ids)
        Parameter : columnsList : columns to include
        Returns : MultiIndexed df, with index : eventInternal, clus2D_id
        """
        return ak.to_dataframe(
            self.array[columnsList],
            levelname=lambda i : {0 : "eventInternal", 1:"clus2D_id"}[i]
        )
    
    @cached_property
    def clusters2D(self) -> pd.DataFrame:
        """
        Builds a pandas DataFrame holding all 2D cluster information (without any rechit info)
        MultiIndex : eventInternal  clus2D_id
        Columns :  beamEnergy	clus2D_x	clus2D_y	clus2D_z	clus2D_energy	clus2D_layer	clus2D_size	clus2D_rho	clus2D_delta	clus2D_pointType
        """
        return self.clusters2D_custom(["beamEnergy", "clus2D_x", "clus2D_y", "clus2D_z", "clus2D_energy", "clus2D_layer", "clus2D_size",
                "clus2D_rho", "clus2D_delta", "clus2D_pointType"])

    @cached_property
    def clusters2D_withNearestHigher(self) -> pd.DataFrame:
        """
        Builds a pandas DataFrame holding all 2D cluster information (without any rechit info)
        MultiIndex : eventInternal  clus2D_id
        Columns :  beamEnergy	clus2D_x	clus2D_y	clus2D_z	clus2D_energy	clus2D_layer	clus2D_size	clus2D_rho	clus2D_delta clus2D_nearestHigher clus2D_pointType
        """
        return self.clusters2D_custom(["beamEnergy", "clus2D_x", "clus2D_y", "clus2D_z", "clus2D_energy", "clus2D_layer", "clus2D_size",
                "clus2D_rho", "clus2D_delta", "clus2D_nearestHigher", "clus2D_pointType"])

    
    def clusters2D_with_rechit_id(self, clusters2D_columns:list[str]=["beamEnergy", "clus2D_layer", "clus2D_rho", "clus2D_delta", "clus2D_pointType"]) -> pd.DataFrame:
        """
        Parameters : - clusters2D_columns : list of columns to include 
        MultiIndex : eventInternal	clus2D_id	rechits_id
        Columns : clus2D_idxs and all clusters2D_columns [beamEnergy	clus2D_layer	clus2D_size		clus2D_rho	clus2D_delta	clus2D_pointType]
        """
        columns = set(clusters2D_columns)
        columns.add("clus2D_idxs")
        return (
            ak.to_dataframe(
                self.array[list(columns)], 
                levelname=lambda i : {0 : "eventInternal", 1:"clus2D_id", 2:"rechit_internal_id"}[i]
            )
            # rechit_internal_id is an identifier counting rechits in each 2D cluster (it is NOT the same as rechits_id, which is unique per event, whilst rechit_internal_id is only unique per 2D cluster)
            .reset_index(level="rechit_internal_id", drop=True)
            .rename(columns={"clus2D_idxs" : "rechits_id"})
        )

    #@memoized_method(maxsize=None) # For now not cached since only called once, in clusters3D_merged_rechits_custom
    def clusters2D_merged_rechits(self, rechitsColumns:frozenset[str], useCachedRechits=True, clusters2D_columns:frozenset[str]=frozenset(["beamEnergy", "clus2D_layer", "clus2D_rho", "clus2D_delta", "clus2D_pointType"])) -> pd.DataFrame:
        """
        Merge clusters2D with rechits
        Parameters : 
            - rechitsColumns: columns to select from rechits Dataframe (as a frozenset so it can work with functools.cache)
            - useCachedRechits : if True, will use self.rechits. If False, will load when called the rechits from file with only the requested rechitsColumns
            - clusters2D_columns : columns to select from clusters2D_with_rechits dataframe
        MultiIndex : eventInternal	clus2D_id	
        Columns : beamEnergy	clus2D_layer	clus2D_rho	clus2D_delta clus2D_idxs	clus2D_pointType 
            and from rechits : rechits_id rechits_x	rechits_y	rechits_z	rechits_energy	rechits_layer	rechits_rho	rechits_delta	rechits_pointType
        beamEnergy_from_rechits is just a duplicate of beamEnergy
        """
        if useCachedRechits:
            rechits_df = self.rechits[list(rechitsColumns)]
        else:
            rechits_df = self.rechits_custom(list(rechitsColumns))
        return pd.merge(
            self.clusters2D_with_rechit_id(list(clusters2D_columns)),     # Left : clusters2D with rechits_id column (one row per rechit of each 2D cluster)
            rechits_df, # Right : rechits
            how='inner',                        # Do an inner join (keeps only rechits that are in a 2D cluster, ie drop outliers). Left should be identical. 
            # Outer and right would include also rechits that are outliers (not associated to any cluster)
            left_on=["eventInternal", "rechits_id"],   # Left ; join on columns : eventInternal, rechits_id
            right_index=True,                   # Right : join on index, ie eventInternal, rechits_id
            suffixes=(None, "_from_rechits"), # Deal with columns present on both sides (none normally) :
            #        take the column from left with same name, the one from right renamed beamEnergy_from_rechits (they should be identical anyway)
            # Disable cross-cjeck for performance
            #validate="one_to_one"               # Cross-check :  Make sure there are no weird things (such as duplicate ids), should not be needed
        )

    @cached_property
    def clusters2D_totalEnergyPerEvent(self) -> pd.DataFrame:
        """ Computer per event the total clustered energy by CLUE2D
        Index : eventInternal
        Columns : beamEnergy clus2D_energy_sum clus2D_energy_sum_fractionOfSynchrotronBeamEnergy
        """
        return (
            self.clusters2D[["clus2D_energy"]].groupby(by=['eventInternal']).agg(
                clus2D_energy_sum=pd.NamedAgg(column="clus2D_energy", aggfunc="sum"),
            )
            .pipe(self.join_divideByBeamEnergy, "clus2D_energy_sum")
        )
    
    def get_clusters2D_perLayerInfo(self, withBeamEnergy=True) -> pd.DataFrame:
        """
        Compute per event and per layer the total 2D-cluster energies (and the same as a fraction of beam energy) and the number of 2D clusters
        Parameter : withBeamEnergy : whether to add beamEnergy column
        Index : eventInternal, clus2D_layer
        Column : [beamEnergy,] clus2D_energy_sum clus2D_count [clus2D_energy_sum_fractionOfSynchrotronBeamEnergy]
        """
        
        df = (self.clusters2D[["clus2D_layer", "clus2D_energy"]]
            .groupby(by=['eventInternal', 'clus2D_layer'])
            .agg(
                clus2D_energy_sum=pd.NamedAgg(column="clus2D_energy", aggfunc="sum"),
                clus2D_count=pd.NamedAgg(column="clus2D_energy", aggfunc="count")
            )
        )
        if withBeamEnergy:
            return self.join_divideByBeamEnergy(df, "clus2D_energy_sum")
        else:
            return df

    @property
    def get_clusters2D_perLayerInfo_allLayers(self) -> pd.DataFrame:
        """
        Same as get_clusters2D_perLayerInfo, except that layers with no 2D clusters have a row added (with zero energy)
        This is intended for profile histograms, to get the correct mean values.
        
        Parameter : withBeamEnergy : whether to add beamEnergy column
        Index : eventInternal, clus2D_layer
        Column : beamEnergy, synchrotronBeamEnergy clus2D_energy_sum clus2D_count clus2D_energy_sum_fractionOfSynchrotronBeamEnergy
        """

        df = self.get_clusters2D_perLayerInfo(withBeamEnergy=False)
        # We build the cartesian product event * layer
        newIndex = pd.MultiIndex.from_product([df.index.levels[0], df.index.levels[1]])


        return (
            # Reindex the dataframe, this will create new rows as needed, filled with zeros
            # Make sure only columns where 0 makes sense are included (not beamEnergy !)
            df.reindex(newIndex, fill_value=0)
            
            # Put beamEnergy back
            .pipe(self.join_divideByBeamEnergy, colName="clus2D_energy_sum")
        )
        

    @property
    def clusters2D_sumClustersOnLayerWithMaxClusteredEnergy(self) -> pd.DataFrame:
        """
        For each event, find the layer with the maximum clustered energy (of 2D clusters) and give the sum of 2D clustered energy on this layer (and the nb of 2D clusters)
        Columns : eventInternal clus2D_layer beamEnergy	clus2D_energy_sum	clus2D_count
        Note : resulting dataframe is not sorted by eventInternal number
        """
        # Old method : using idxmax : very slow
            # Start from clusters2D_maxEnergyPerLayer and build the index 
            # First group by event
            # Then compute the layer nb of the maximum value clus2D_energy_sum per event
            # index = self.get_clusters2D_perLayerInfo(withBeamEnergy=False)[["clus2D_energy_sum"]].groupby(by=["eventInternal"]).idxmax()
            # Index is a Series with : Index=eventInternal, Column=(eventInternal, layer) (as a tuple)

            # Apply the index, this will select for each event, only rows with the right layer
            #return self.get_clusters2D_perLayerInfo().loc[index.clus2D_energy_sum].reset_index(level=["clus2D_layer"])
        
        # New method : uses sorting and drop_duplicates (at least an order of magnitude faster, but event nb are not sorted)
        return (self.get_clusters2D_perLayerInfo()
            .reset_index() # So eventInternal is a column
            .sort_values("clus2D_energy_sum", ascending=False) # Sort descending on cluster 2D energy sum
            .drop_duplicates("eventInternal", keep="first") # Drop duplicate events, will keep only the first row ie the one with highest energy sum
        )

    @cached_property
    def clusters3D(self) -> pd.DataFrame:
        """
        MultiIndex : eventInternal   clus3D_id
        Columns :    clus3D_x	clus3D_y	clus3D_z	clus3D_energy	clus3D_layer clus3D_size
        """
        return (ak.to_dataframe(self.array[
            ["beamEnergy", "clus3D_x", "clus3D_y", "clus3D_z", "clus3D_energy", "clus3D_size"]
            ], 
            levelname=lambda i : {0 : "eventInternal", 1:"clus3D_id"}[i])
        )

    @property
    def clusters3D_countPerEvent(self) -> pd.DataFrame:
        """ Count 3D clusters for each event
        Index : eventInternal
        Columns : beamEnergy clus3D_count """
        return self.clusters3D.groupby("eventInternal").agg(
            beamEnergy=pd.NamedAgg("beamEnergy", "first"),
            clus3D_count=pd.NamedAgg("beamEnergy", "count")
        )

    @cached_property
    def clusters3D_largestClusterIndex(self) -> pd.MultiIndex:
        """
        Compute for each event, the index of the 3D cluster with the largest clustered energy (clus3D_energy)
        (in case of equality returns the one that comes later in the dataset)
        Returns a MultiIndex (eventInternal, clus3D_id), to be used with loc (on a df indexed by (eventInternal, clus3D_id) ):
        ex : clusters3D.loc[clusters3D_largestClusterIndex]
        """
        return pd.MultiIndex.from_frame(# Make a multiindex out of it
            self.clusters3D
            [["clus3D_energy"]] # Dataframe is index=(eventInternal, clus3D_id), columns=clus3D_energy
            .reset_index() # So we can sort_values on eventInternal
            .sort_values(["eventInternal", "clus3D_energy"], ascending=True) # Ascending so that event nb does not get unsorted
            .drop_duplicates("eventInternal", keep="last") # Keep for each event only the clus3D_id with highest clus3D_energy, which is the last one in the list (ascending=True)
            .drop(columns="clus3D_energy")
        )

        # Old, slow version: 
        #return self.clusters3D.groupby(["eventInternal"])["clus3D_energy"].idxmax()
    
    @cached_property
    def clusters3D_largestClusterIndex_fast(self) ->Iterable[tuple[Any, ...]]:
        """
        Compute for each event, the index of the 3D cluster with the largest clustered energy (clus3D_energy)
        (in case of equality returns the one that comes later in the dataset)
        Returns an iterator over tuples (eventInternal, clus3D_id) to be used as :
        df[df.index.isin(comp.clusters3D_largestClusterIndex_fast)]
        Which is much faster for large dataframes than using df.loc[clusters3D_largestClusterIndex]
        """
        # Note : list is needed since Pandas v2 as it calls len() on iterable, which does not work for zip objects (TypeError)
        return list(# Make a multiindex out of it
            self.clusters3D
            [["clus3D_energy"]] # Dataframe is index=(eventInternal, clus3D_id), columns=clus3D_energy
            .reset_index() # So we can sort_values on eventInternal
            .sort_values(["eventInternal", "clus3D_energy"], ascending=True) # Ascending so that eventInternal nb does not get unsorted
            .drop_duplicates("eventInternal", keep="last") # Keep for each event only the clus3D_id with highest clus3D_energy, which is the last one in the list (ascending=True)
            .drop(columns="clus3D_energy")
            .itertuples(index=False, name=None)
        )

    @property
    def clusters3D_largestCluster(self) -> pd.DataFrame:
        """ 
        Same as clusters3D but only, for each event, with the 3D cluster with highest energy 
        Index : eventInternal
        Columns : beamEnergy, clus3D_* 
        """
        return self.clusters3D.loc[self.clusters3D_largestClusterIndex]

    #Don't cache this as unlikely to be needed again after caching clusters3D_merged_2D
    def clusters3D_with_clus2D_id(self, extraColumns:list[str]=[]) -> pd.DataFrame:
        """ Makes a Dataframe with clusters3D info and 2D clusters ID (to be joined later)
        Param : extraColumns : extra columns to add beside clus3D_energy	clus3D_layer clus3D_size clus2D_id
        MultiIndex : eventInternal  clus3D_id
        Columns : 	clus3D_energy	clus3D_layer clus3D_size clus2D_id   # clus3D_x	clus3D_y	clus3D_z
        """
        return (ak.to_dataframe(
            self.array[list(set(["beamEnergy", "clus3D_energy", "clus3D_size", "clus3D_idxs"]).union(extraColumns))],
            levelname=lambda i : {0 : "eventInternal", 1:"clus3D_id", 2:"clus2D_internal_id"}[i]
        )
        # clus2D_internal_id is an identifier counting 2D clusters in each 3D cluster (it is NOT the same as clus2D_id, which is unique per event, whilst clus2D_internal_id is only unique per 3D cluster)
        .reset_index(level="clus2D_internal_id", drop=True)
        .rename(columns={"clus3D_idxs" : "clus2D_id"})
        )
        
    @cached_property
    def clusters3D_merged_2D(self) -> pd.DataFrame:
        """
        Merge the dataframe clusters3D_with_clus2D_id_df with clusters2D
        Param : clusters3D_with_clus2D_id_df : the dataframe to use for left param of join
        If None, uses self.clusters3D_with_clus2D_id
        
        Returns : 
        MultiIndex : eventInternal	clus3D_id	clus2D_internal_id
        Columns : beamEnergy	clus3D_energy	clus3D_size	clus2D_id		clus2D_x	clus2D_y	clus2D_z	clus2D_energy	clus2D_layer	clus2D_rho	clus2D_delta	clus2D_pointType
        as well as beamEnergy_from_2D_clusters which is just a duplicate of beamEnergy
        """
        return self.clusters3D_merged_2D_custom(self.clusters3D_with_clus2D_id(), self.clusters2D)

    def clusters3D_merged_2D_custom(self, clusters3D_with_clus2D_id_df:pd.DataFrame, clusters2D_df:pd.DataFrame) -> pd.DataFrame:
        """ Same as above but use custom dataframes for 3D part (must have eventInternal, clus2D_id) and 2D part (must have index (eventInternal, clus2D_id))"""
        return pd.merge(
            clusters3D_with_clus2D_id_df.reset_index(), # Left
            clusters2D_df,                # Right
            how='inner',                    # Inner join (the default). Keeps only 2D clusters that are associated to a 3D cluster (ie drop outliers)
            # Outer and right would include also rechits that are outliers (not associated to any cluster)
            left_on=["eventInternal", "clus2D_id"],  # Left : Join on eventInternal nb and ID of 2D cluster per event
            right_index=True,                  # Right : Join on MultiIndex, ie on eventInternal and clus2D_id
            suffixes=(None, "_from_2D_clusters"), # Deal with columns present on both sides (currently only beamEnergy) :
            #        take the column from left with same name, the one from right renamed beamEnergy_from_2D_clusters (they should be identical anyway)
            validate="one_to_one"               # Cross-check :  Make sure there are no weird things (such as duplicate ids), should not be needed
        )#.droplevel(level="clus2D_internal_id") # remove the useless clus2D_internal_id column
    
    @cached_property
    def clusters3D_merged_2D_impact(self) -> pd.DataFrame:
        return self.clusters3D_merged_2D_impact_custom(self.clusters3D_merged_2D)

    def clusters3D_merged_2D_impact_custom(self, clusters3D_merged_2D_df:pd.DataFrame|None = None) -> pd.DataFrame:
        """
        Merge clusters3D_merged_2D with impact dataframe, to get impact info for all 2D clusters members of a 3D cluster
        Also creates clus2D_diff_impact_x and clus2D_diff_impact_y columns holding the difference between 2D cluster position and extrapolated track impact position on layer
        """
        if clusters3D_merged_2D_df is None:
            clusters3D_merged_2D_df = self.clusters3D_merged_2D
        merged_df = pd.merge(
            # Left : previously merged dataframe
            clusters3D_merged_2D_df,

            #Right : impact df (indexed by eventInternal and layer)
            self.impact, 

            # Map event on both sides
            # Map layer of 2D cluster with layer of impact computation
            left_on=("eventInternal", "clus2D_layer"),
            right_on=("eventInternal", "layer")
        ).set_index(["eventInternal", "clus3D_id"]) # Add clus3D_id to the index (so we can select only main 3D clusters from clusters3D_largestClusterIndex)

        merged_df["clus2D_diff_impact_x"] = merged_df["clus2D_x"] - merged_df["impactX"]
        merged_df["clus2D_diff_impact_y"] = merged_df["clus2D_y"] - merged_df["impactY"]
        return merged_df

    
    def clusters3D_firstLastLayer(self, clusters3D_merged_2D_df:pd.DataFrame, columnsToKeep=["beamEnergy", "clus3D_energy", "clus3D_size"]) -> pd.DataFrame:
        """
        For each 3D cluster, compute the first and last layer numbers of contained 2D clusters 
        Param : 
         - clusters3D_merged_2D_df : dataframe to consider, should be a subset of self.clusters3D_merged_2D
         - columnsToKeep : list of columns that should be kept during grouping (should be 3D-cluster or event related columns)
        Returns : 
        MultiIndex : eventInternal, clus3D_id
        Columns : clus2D_minLayer, clus2D_maxLayer and columnsToKeep (beamEnergy, clus3D_energy, clus3D_size by default)

        Should be used as :
        self.clusters3D_with_clus2D_id[.pipe(clusters3D_filterLargestCluster)].pipe(clusters3D_merged_2D).pipe(self.clusters3D_firstLastLayer)
        """
        agg_kwargs = {colName : pd.NamedAgg(column=colName, aggfunc="first") for colName in columnsToKeep}
        agg_kwargs.update(dict(
            clus2D_minLayer=pd.NamedAgg(column="clus2D_layer", aggfunc="min"),
            clus2D_maxLayer=pd.NamedAgg(column="clus2D_layer", aggfunc="max"),
        ))

        return (clusters3D_merged_2D_df#[["eventInternal", "clus3D_id", "beamEnergy", "clus3D_energy", "clus3D_size", "clus2D_layer"]]
            .groupby(["eventInternal", "clus3D_id"])
            .agg(**agg_kwargs)
        )

    # def clusters3D_indexOf3DClustersPassingMinNumLayerCut(self, minNumLayerCluster):
    #     df = self.clusters3D_merged_2D[["clus2D_layer"]].groupby(["eventInternal", "clus3D_id"]).agg(
    #         clus2D_layer_min=pd.NamedAgg(column="clus2D_layer", aggfunc="min"),
    #         clus2D_layer_max=pd.NamedAgg(column="clus2D_layer", aggfunc="max"),
    #     )
    #     df["clus3D_numLayers"] = df["clus2D_layer_max"] - df["clus2D_layer_min"]+1
    #     return df["clus3D_numLayers"] < minNumLayerCluster
    
    @cached_property
    def clusters3D_energyClusteredPerLayer(self) -> pd.DataFrame:
        """ Compute total 2D clustered energy per layer for each 3D cluster 
        MultiIndex : eventInternal	clus3D_id	clus2D_layer
        Columns : beamEnergy clus3D_size clus3D_energy	clus2D_energy_sum
        """
        return (
            self.clusters3D_merged_2D[["eventInternal", "clus3D_id", "beamEnergy", "clus3D_energy", "clus3D_size", "clus2D_energy", "clus2D_layer"]]

            # For each event, cluster 3D and layer, sum clus2D_energy
            .groupby(by=["eventInternal", "clus3D_id", "clus2D_layer"]).agg(
                beamEnergy=pd.NamedAgg(column="beamEnergy", aggfunc="first"),
                clus3D_size=pd.NamedAgg(column="clus3D_size", aggfunc="first"),
                clus3D_energy=pd.NamedAgg(column="clus3D_energy", aggfunc="first"),
                clus2D_energy_sum=pd.NamedAgg(column="clus2D_energy", aggfunc="sum")
            )
        )
    
    @cached_property
    def clusters3D_energyClusteredPerLayer_allLayers(self) -> pd.DataFrame:
        """ Same as clusters3D_energyClusteredPerLayer but inserts rows with zeroes for layers wich have no layer clusters (to compute profile histograms properly)
        MultiIndex : eventInternal, clus3D_id
        Columns : clus3D_size clus2D_layer	clus2D_energy_sum	beamEnergy	synchrotronBeamEnergy	clus2D_energy_sum_fractionOfsynchrotronBeamEnergy
        """
        df = (self.clusters3D_merged_2D[["eventInternal", "clus3D_id", "clus3D_energy", "clus3D_size", "clus2D_energy", "clus2D_layer"]]
        # For each event, cluster 3D and layer, sum clus2D_energy
            .groupby(by=["eventInternal", "clus3D_id", "clus2D_layer"]).agg(
                clus2D_energy_sum=pd.NamedAgg(column="clus2D_energy", aggfunc="sum")
            )
        )
        # Make list of tuples holding (eventInternal, clus3D_id)
        tuples_event_clus3D_id:list[tuple[int, int]] = df.index.droplevel("clus2D_layer").drop_duplicates().to_list()

        # Makes list of all unique values of layer numbers
        layers:list[int] = df.index.levels[2].to_list() 

        # Build the new MultiIndex with all layers, from a list of 3-tuples
        newIndex = pd.MultiIndex.from_tuples(
            ((eventInternal, clus3D_id, layer) for eventInternal, clus3D_id in tuples_event_clus3D_id for layer in layers),
            names=["eventInternal", "clus3D_id", "clus2D_layer"]
        )

        return (
            # Reindex the dataframe, this will create new rows as needed, filled with zeros
            # Make sure only columns where 0 makes sense are included (not beamEnergy !)
            # Also it does not make sense for clus3D_* properties
            df.reindex(newIndex, fill_value=0)
            
            # Put beamEnergy back
            .pipe(self.join_divideByBeamEnergy, colName="clus2D_energy_sum")

            # Put clus3D_size back
            .join(
                self.clusters3D[["clus3D_size"]],
                on=["eventInternal", "clus3D_id"]
            )

            .reset_index("clus2D_layer")
        )

    @cached_property
    def clusters3D_layerWithMaxClusteredEnergy(self) -> pd.DataFrame:
        """ Select layer with maximum 2D clustered energy in each 3D cluster 
        MultiIndex : eventInternal clus3D_id
        Columns : layer_with_max_clustered_energy	beamEnergy	clus3D_size	clus3D_energy	clus2D_energy_sum (sum of 2D clustered energy of this 3D cluster in the layer with max energy) """
        # for each event and 3D cluster find the layer with the maximum clus2D_energy_sum
        return (self.clusters3D_energyClusteredPerLayer
            .reset_index()
            .sort_values("clus2D_energy_sum", ascending=False)
            .drop_duplicates(["eventInternal", "clus3D_id"], keep="first") # Drop duplicates : will keep only highest value of clus2D_energy_sum for each event, clus3D_id
        ).rename(columns={"clus2D_layer":"layer_with_max_clustered_energy"})

    @cached_property
    def clusters3D_impact_usingLayerWithMax2DClusteredEnergy(self):
        """ Add a column to clusters3D with the difference between clus3D_x and impactX, impactX being computed on the layer with maximum 2D clustered energy (of each cluster 3D)
        Note : eventInternal is not in index and dataframe is not sorted (neither on eventInternal nor on clus3D_id)
        Columns : eventInternal clus3D_id	beamEnergy	clus3D_x	clus3D_y	clus3D_z	clus3D_energy	clus3D_size	layer_with_max_clustered_energy	impactX	impactY	clus3D_diff_impact_x clus3D_diff_impact_y
        """
        # Merge with impact
        df_withImpact = pd.merge(
            # Left
            self.clusters3D_layerWithMaxClusteredEnergy,
            # Right :
            self.impact,

            how='left', # Left join so we preserve all 3D clusters
            # Map event and layer
            left_on=["eventInternal", "layer_with_max_clustered_energy"],
            right_on=["eventInternal", "layer"]
        )

        # We add back clus3D_x and clus3D_y by joining with clusters3D :
        final_df = pd.merge(
            df_withImpact,
            self.clusters3D[["clus3D_x", "clus3D_y"]],
            on=["eventInternal", "clus3D_id"]
        )

        # Make the substractions : 
        final_df["clus3D_diff_impact_x"] = final_df["clus3D_x"] - final_df["impactX"]
        final_df["clus3D_diff_impact_y"] = final_df["clus3D_y"] - final_df["impactY"]
        return final_df

    @cached_property
    def clusters3D_merged_rechits(self):
        """ Merge clusters3D dataframe with clusters2D then rechits 
        Index : eventInternal, clus3D_id, rechits_layer, rechits_id
        Columns : beamEnergy	clus3D_energy	clus3D_size	clus2D_id	rechits_x	rechits_y	rechits_energy"""
        return self.clusters3D_merged_rechits_custom(["rechits_x", "rechits_y", "rechits_energy"])
    
    def clusters3D_merged_rechits_custom(self, listOfColumns):
        """ Merge clusters3D dataframe with clusters2D then rechits 
        Parameters : listOfColumns = iterable of column names to take from self.rechits (rechits_id and rechits_layer are always included)
        """
        # First merge everything to get rechits information
        return (
            pd.merge(
                self.clusters3D_with_clus2D_id().reset_index("clus3D_id"), # reset clus3D_id index so it does not get lost in the merge
                # rechits_id is Index in rechits df
                # Note the comma, to avoid making a set with all letters of rechits_layer
                self.clusters2D_merged_rechits(frozenset(listOfColumns).union(("rechits_layer",))).drop(columns="beamEnergy"), 
                on=["eventInternal", "clus2D_id"]
            )
            # To assign series later, we need a unique multiindex so we use rechits_id
            # Also set clus3D_id for later groupby
            .set_index(["clus3D_id", "rechits_layer", "rechits_id"], append=True)
        )

    def computeBarycenter(self, df:pd.DataFrame, groupbyColumns:list[str], thresholdW0=parameters.thresholdW0):
        """ Compute barycenter of 3D clusters.
        The barycenter is computed, in each group given by groupByColumns, using rechits log weighted positions.
        Uses thresholdW0 (default value from parameters.py). The total energy sum used for the log is the sum of energies of all rechits in
        the considered group
        Parameters : 
            - df : DataFrame with rechits_{x, y, energy} plus necessary groupBy columns in index
            - groupByColumns : list of columns to group by when computing barycenter (should include event and rechits_layer, and eventually clus2D_id or clus3D_id)
            - thresholdW0 : parameter of log-weighting of energies
        Returns dataframe with :
        Index : whatever what is input df
        Columns : 
           - rechits_x_barycenter, rechits_y_barycenter : barycenters per layer and 3D cluster
           - rechits_energy_sumPerLayer : sum of all rechits energies on a layer (simple sum, no log weights)
           - rechits_countPerLayer of rechits in the layer (and in 3D cluster, event)
        """
        # Compute Wi = max(0: thresholdW0 + ln(E / sumE)) where sumE is the sum of rechits energies in same layer and belonging to same 3D cluster
        # Use assign to make a copy
        # The next two lines are a significant performance bottleneck (lots of rechits)
        df = df.assign(rechit_energy_logWeighted=(
            (thresholdW0 + np.log(df.rechits_energy / df.rechits_energy.groupby(by=groupbyColumns).sum()))
            .clip(lower=0.) # Does max(0; ...)
        ))

        ### Start computing barycenter = ( sum_i Xi * Wi ) / ( sum Wi )
        # Compute x * Wi
        df["rechits_x_times_logWeight"] = df["rechits_x"]*df["rechit_energy_logWeighted"]
        df["rechits_y_times_logWeight"] = df["rechits_y"]*df["rechit_energy_logWeighted"]

        # Make sums : 
        df_groupedPerLayer = df.groupby(by=groupbyColumns).agg(
            # sum Wi
            rechits_logWeight_sumPerLayer=pd.NamedAgg(column="rechit_energy_logWeighted", aggfunc="sum"),
            # Sum Xi*Wi
            rechits_x_times_logWeight_sumPerLayer=pd.NamedAgg(column="rechits_x_times_logWeight", aggfunc="sum"),
            rechits_y_times_logWeight_sumPerLayer=pd.NamedAgg(column="rechits_y_times_logWeight", aggfunc="sum"),

            # Sum energy per layer (not for barycenter, but for normalization)
            rechits_energy_sumPerLayer=pd.NamedAgg(column="rechits_energy", aggfunc="sum"),
            # Count hits in layer
            rechits_countPerLayer=pd.NamedAgg(column="rechits_energy", aggfunc="count")
        )
        # Barycenter positions : ( sum_i Xi * Wi ) / ( sum Wi )
        df_groupedPerLayer["rechits_x_barycenter"] = df_groupedPerLayer["rechits_x_times_logWeight_sumPerLayer"]/df_groupedPerLayer["rechits_logWeight_sumPerLayer"]
        df_groupedPerLayer["rechits_y_barycenter"] = df_groupedPerLayer["rechits_y_times_logWeight_sumPerLayer"]/df_groupedPerLayer["rechits_logWeight_sumPerLayer"]

        return df_groupedPerLayer[["rechits_x_barycenter", "rechits_y_barycenter", "rechits_energy_sumPerLayer", "rechits_countPerLayer"]]

    @memoized_method(maxsize=None)
    def clusters3D_computeBarycenter(self, thresholdW0=parameters.thresholdW0):
        """ Compute barycenter of 3D clusters.
        The barycenter is computed, for each event, cluster 3D and layer, using rechits log weighted positions.
        Uses thresholdW0 from parameters.py. The total energy sum used for the log is the sum of energies of all rechits in
        the considered 3D cluster *on the same layer* (as is done by CLUE2D for computing layer cluster positions,
        except possibly considering more than one layer cluster per layer in case the 3D cluster has more than one).
        Returns dataframe with :
        Index : eventInternal clus3D_id rechits_layer
        Columns : 
           - rechits_x_barycenter, rechits_y_barycenter : barycenters per layer and 3D cluster
           - rechits_energy_sumPerLayer : sum of all rechits energies on a layer (simple sum, no log weights)
           - rechits_countPerLayer of rechits in the layer (and in 3D cluster, event)
        """
        return self.computeBarycenter(self.clusters3D_merged_rechits, groupbyColumns=["eventInternal", "clus3D_id", "rechits_layer"], thresholdW0=thresholdW0)

    @cached_property
    def clusters3D_rechits_distanceToBarycenter_energyWeightedPerLayer(self):
        """ Compute, for all rechits in a 3D cluster, the distance of the rechit position to the barycenter.
        The barycenter is computed, for each event, cluster 3D and layer, using rechits log weighted positions.
        Uses thresholdW0 from parameters.py. The total energy sum used for the log is the sum of energies of all rechits in
        the considered 3D cluster *on the same layer* (as is done by CLUE2D for computing layer cluster positions,
        except possibly considering more than one layer cluster per layer in case the 3D cluster has more than one).

        Index : eventInternal, clus3D_id
        Colums : beamEnergy rechits_layer	clus3D_energy	clus3D_size	clus2D_id	rechits_x	rechits_y	rechits_distanceToBarycenter
        rechits_id rechits_energy	rechit_energy_logWeighted	rechits_x_times_logWeight	rechits_y_times_logWeight
        """

        # Faster (factor 2) method with eval:
        return (
            self.clusters3D_merged_rechits # Start from all rechits
            [["beamEnergy", "clus3D_size", "rechits_x", "rechits_y", "rechits_energy"]]
            .join(self.clusters3D_computeBarycenter()) # Broadcast per-layer informations back into rechit-level dataframe
            # Compute new columns. Using eval is slightly faster than doing it in Python (when df is very large, factor 2 gain in time)
            .eval("""
        rechits_distanceToBarycenter = sqrt((rechits_x - rechits_x_barycenter)**2 + (rechits_y - rechits_y_barycenter)**2)
        rechits_energy_EnergyFractionNormalized = rechits_energy / rechits_energy_sumPerLayer
        rechits_1_over_rechit_count = 1./rechits_countPerLayer
        """
            )
            .drop(columns=["rechits_x", "rechits_y", "rechits_x_barycenter", "rechits_y_barycenter", "rechits_energy_sumPerLayer", "rechits_countPerLayer"])
            .reset_index(["rechits_layer", "rechits_id"]) # Drop rechits_layer index as needed for filling histograms
        )

    @cached_property
    def rechits_distanceToImpact(self):
        df = pd.merge(
            (self.rechits[["beamEnergy", "rechits_layer", "rechits_x", "rechits_y", "rechits_energy"]]
                .set_index("rechits_layer", append=True)
                .reorder_levels(["eventInternal", "rechits_layer", "rechits_id"])
            ),
            self.impact,
            left_on=["eventInternal", "rechits_layer"],
            right_index=True
        )
        
        grouped_df = df.rechits_energy.groupby(by=["eventInternal", "rechits_layer"]).agg(
            rechits_energy_sumPerLayer="sum",
            rechits_countPerLayer="count",
        )

        return (df
            .join(grouped_df)
            .eval("""
        rechits_distanceToImpact = sqrt((rechits_x - impactX)**2 + (rechits_y - impactY)**2)
        rechits_energy_EnergyFractionNormalized = rechits_energy / rechits_energy_sumPerLayer
        rechits_1_over_rechit_count = 1./rechits_countPerLayer
        """
            )
            .drop(columns=["rechits_x", "rechits_y", "impactX", "impactY", "rechits_energy_sumPerLayer", "rechits_countPerLayer"])
            .reset_index(["rechits_layer", "rechits_id"])
        )

    @cached_property
    def clusters2D_rechits_distanceToImpact(self):
        df = pd.merge(
            (self.clusters2D_merged_rechits(frozenset(["beamEnergy", "rechits_layer", "rechits_x", "rechits_y", "rechits_energy"]))
                [["beamEnergy", "rechits_layer", "rechits_x", "rechits_y", "rechits_energy"]] # remove all clus2D_*
                .set_index("rechits_layer", append=True)
                .reorder_levels(["eventInternal", "rechits_layer", "clus2D_id"])
            ),
            self.impact,
            left_on=["eventInternal", "rechits_layer"],
            right_index=True
        )
        
        grouped_df = df.rechits_energy.groupby(by=["eventInternal", "rechits_layer"]).agg(
            rechits_energy_sumPerLayer="sum",
            rechits_countPerLayer="count",
        )

        return (df
            .join(grouped_df)
            .eval("""
        rechits_distanceToImpact = sqrt((rechits_x - impactX)**2 + (rechits_y - impactY)**2)
        rechits_energy_EnergyFractionNormalized = rechits_energy / rechits_energy_sumPerLayer
        rechits_1_over_rechit_count = 1./rechits_countPerLayer
        """
            )
            .drop(columns=["rechits_x", "rechits_y", "impactX", "impactY", "rechits_energy_sumPerLayer", "rechits_countPerLayer"])
            .reset_index(["rechits_layer"])
        )

    @cached_property
    def clusters3D_rechits_distanceToImpact(self):
        """ Merges clusters3D with rechits, then computes distance to impact for each rechit
        Index : eventInternal	clus3D_id	rechits_id
        Columns : rechits_layer	beamEnergy	clus3D_size	rechits_energy	rechits_distanceToImpact	rechits_energy_EnergyFractionNormalized	rechits_1_over_rechit_count
        """
        df = pd.merge(
            self.clusters3D_merged_rechits[["beamEnergy", "clus3D_size", "rechits_x", "rechits_y", "rechits_energy"]],
            self.impact,
            left_on=["eventInternal", "rechits_layer"],
            right_index=True
        )
        
        grouped_df = df.rechits_energy.groupby(["eventInternal", "clus3D_id", "rechits_layer"]).agg(
            rechits_energy_sumPerLayer="sum",
            rechits_countPerLayer="count",
        )
        
        return (df
            .join(grouped_df)
            .eval("""
        rechits_distanceToImpact = sqrt((rechits_x - impactX)**2 + (rechits_y - impactY)**2)
        rechits_energy_EnergyFractionNormalized = rechits_energy / rechits_energy_sumPerLayer
        rechits_1_over_rechit_count = 1./rechits_countPerLayer
        """
            )
            # Old way to compute area normalization (now it is done after filling)
            # rechits_energy_AreaNormalized = rechits_energy_EnergyFractionNormalized / ( @math.pi * (2 * rechits_distanceToImpact * @dr  + @dr*@dr))
            .drop(columns=["rechits_x", "rechits_y", "impactX", "impactY", "rechits_energy_sumPerLayer", "rechits_countPerLayer"])
            .reset_index(["rechits_layer", "rechits_id"])
        )

    @property
    def clusters3D_PCA(self):
        """ Compute PCA on layer clusters of each 3D cluster
        Returns pd.Series, indexed by eventInternal, clus3D_id, with a numpy 3-vector holding main PCA direction 
        """
        return self.clusters3D_merged_2D.groupby(["eventInternal", "clus3D_id"]).apply(Cluster3D_PCA)

    def clusters3D_PCA_angleWithImpact(self, clusters2D_impact_df:pd.DataFrame, name:str) -> pd.Series:
        """ Compute PCA on layer clusters of each 3D cluster 
        Parameters : 
        - clusters2D_impact_df : dataframe with : Index : eventInternal, clus3D_id; Columns : "clus2D_x", "clus2D_y", "clus2D_z", "clus2D_energy", "clus2D_layer", "impactX", "impactY"
            holding only layer clusters that need to be included in PCA
        - name to insert 
        Returns Dataframe indexed by eventInternal, clus3D_id, with columns :
        - clus3D_pca_impact_NAME_angle : the angle (in [0, pi/2] range) between PCA estimate and DWC track
        - clus3D_pca_impact_NAME_angle_x : the angle of the vectors projected in (Oxz) plane (in [-pi/2, pi/2] range, right-handed, angle PCA-> impact)
        - clus3D_pca_impact_NAME_angle_y : same but in (Oyz) plane
        """
        return (clusters2D_impact_df
            .groupby(["eventInternal", "clus3D_id"])
            .apply(partial(computeAngleSeries, name=name))
            #.rename("clus3D_angle_pca_impact")
        )
    
    @cached_property
    def clusters3D_PCA_dataframe(self) -> pd.DataFrame:
        """ Compute PCA shower axis for 3D clusters (selecting those that span at least 3 layers), then compute the angle with the axis from DWC
        Returns a Dataframe indexed by eventInternal, clus3D_id with clusters3D columns as well as : 
        - clus3D_angle_pca_impact_filterLayerSpan : angle where PCA was done with all 2D clusters (only 3D clusters were filtered using layer span)
        - clus3D_angle_pca_impact_filterLayerSpan_cleaned : PCA was done on subset of 2D clusters inside 3D clusters (consider only highest energy 
            2D cluster on each layer, and only layer within )"""
        clusters3D_full_df = (self.clusters3D_merged_2D_impact
            [["clus2D_id",  "clus2D_energy", "clus2D_layer", "clus2D_x", "clus2D_y", "clus2D_z", "impactX", "impactY"]]
        )

        # Compute the filter to get only 3D clusters that span at least 3 layers
        filter_clus3D_layerSpan = (clusters3D_full_df
            .clus2D_layer.groupby(by=["eventInternal", "clus3D_id"]).nunique() >= 3
        )
        # ---- Do the PCA with no cleaning, just the filter for layer span
        pca_filterLayerSpan = (clusters3D_full_df
            .loc[filter_clus3D_layerSpan]
            .pipe(self.clusters3D_PCA_angleWithImpact, "filterLayerSpan")
        )

        # ---- PCA with cleaning of 2D clusters
        # Cleaning, step 1 : Select on each event, cluster3D, layer the 2D cluster with the highest energy
        df_highestLCEnergy = (clusters3D_full_df
            .sort_values(by=["eventInternal", "clus3D_id", "clus2D_layer", "clus2D_energy"], ascending=True)
            .reset_index()
            .drop_duplicates(["eventInternal", "clus3D_id", "clus2D_layer"], keep="last")
        )

        
        df_highestLCEnergy = (df_highestLCEnergy
            .set_index(["eventInternal", "clus3D_id"]) # need indexing for broadcasting back the layer number

            # Select for each event, 3D cluster the 2D cluster with max energy
            # and put the layer number back into the dataframe
            .assign(maxEnergy2DClusterIn3DCluster_layer=(
                df_highestLCEnergy.sort_values(by=["eventInternal", "clus3D_id", "clus2D_energy"])
                    .drop_duplicates(["eventInternal", "clus3D_id"], keep="last")
                    .set_index(["eventInternal", "clus3D_id"])
                    .clus2D_layer
                )
            )
            # Cleaning step 2 : select only layer clusters that are within a distance of the maximum energy layer cluster
            .query("(clus2D_layer < maxEnergy2DClusterIn3DCluster_layer + 16) and "
                "(clus2D_layer > maxEnergy2DClusterIn3DCluster_layer - 11)")
            
            .loc[filter_clus3D_layerSpan] # Filter on layer span
        )
        # Actually compute the PCA
        pca_filterLayerSpan_cleaned2DClusters = (
            df_highestLCEnergy
            .pipe(self.clusters3D_PCA_angleWithImpact, "filterLayerSpan_cleaned")
        )

        #return pca_filterLayerSpan, pca_filterLayerSpan_cleaned2DClusters

        #Put the two series with PCA angles together
        merged_df =  pd.concat(
            [pca_filterLayerSpan, pca_filterLayerSpan_cleaned2DClusters],
            axis="columns", join="outer"
        )
        # Put back 3D cluster information into dataframe (using left join so only selected 3D clusters are put back)
        return pd.merge(
            merged_df,
            self.clusters3D,
            how="left",
            left_index=True, right_index=True
        )

    def clusters3D_intervalHoldingFractionOfEnergy(self, fraction:float, engine:str|None=None, maskLayer:int=None) -> pd.DataFrame:
        """ Compute, for each 3D cluster, the shortest interval [first layer; last layer] that contains at least fraction of the 3D cluster energy
        Parameters :
         - fraction
         - engine : can be python, cython, None (None prefers Cython)
         - maskLayer : if not None, then mask this layer number when computing intervals (also mask for computing total 3D cluster energy for fraction computation)
        Index : eventInternal, clus3D_id
        Columns : intervalFractionEnergy_minLayer	intervalFractionEnergy_maxLayer
        """
        try:
            # Use Cython version in priority
            if engine is None or engine == "cython":
                from .dataframe_cython import computeShortestInterval
                agg_lambda = lambda series : computeShortestInterval(series.to_numpy(), series.index.levels[2].to_numpy(), fraction)
            else:
                raise Exception()
        except Exception as e:
            if engine == "cython":
                raise e
            print("WARNING : (dataframe.py) falling back to python implementation of intervalHoldingFractionOfEnergy, which is very slow. ")
            print("Consider using the cythonized version, by running 'cd Plotting/hists; cythonize -3 -i dataframe_cython.pyx")
            # Python fallback implementation (very slow)
            def computeShortestInterval(series:pd.Series, fraction:float) -> tuple[int, int]:
                """ Compute the actual shortest interval. Params : 
                - series : a Pandas series indexed by eventInternal, clus3D_id, clus2D_layer """
                layers = series.index.levels[2] # List of layers 
                totalEnergyFraction = fraction*np.sum(series)
                bestInterval = (layers[0], layers[-1])
                j = 0
                for i, i_layer in enumerate(layers):
                    j = max(j, i)
                    
                    while np.sum(series[i:j+1]) < totalEnergyFraction:
                        if j >= len(series): # Impossible to find a covering interval at this stage
                            return bestInterval
                            #return pd.DataFrame({"intervalFractionEnergy_minLayer":bestInterval[0], "intervalFractionEnergy_maxLayer":bestInterval[1]}, index=[0])
                        j += 1
                    j_layer = layers[j]
                    if j_layer-i_layer < bestInterval[1] - bestInterval[0]:
                        bestInterval = (i_layer, j_layer)
                
                return bestInterval
                #return pd.DataFrame({"intervalFractionEnergy_minLayer":bestInterval[0], "intervalFractionEnergy_maxLayer":bestInterval[1]}, index=[0])
            agg_lambda = partial(computeShortestInterval, fraction=fraction)


        if maskLayer is None:
            masked_series = self.clusters3D_energyClusteredPerLayer.clus2D_energy_sum
        else:
            masked_series = self.clusters3D_energyClusteredPerLayer[["clus2D_energy_sum"]].reset_index(level="clus2D_layer")
            masked_series = masked_series[masked_series.clus2D_layer != maskLayer].set_index("clus2D_layer", append=True).clus2D_energy_sum
        
        series = (masked_series
            .groupby(["eventInternal", "clus3D_id"])
            .agg(agg_lambda)
        )
        series.name = "intervalFractionEnergy_minMaxLayer"

        # Unpack the tuples
        df = pd.DataFrame(series)
        df["intervalFractionEnergy_minLayer"], df["intervalFractionEnergy_maxLayer"] = zip(*series)
        return df.drop("intervalFractionEnergy_minMaxLayer", axis="columns")

        # return self.clusters3D_energyClusteredPerLayer.clus2D_energy_sum.groupby(["eventInternal", "clus3D_id"]).apply(
        #     partial(computeShortestInterval, fraction=fraction)
        # ).reset_index(level=2, drop=True)
    
    @memoized_method(maxsize=None)
    def clusters3D_intervalHoldingFractionOfEnergy_joined(self, fraction:float, maskLayer:int|None=None) -> pd.DataFrame:
        """ Same as clusters3D_intervalHoldingFractionOfEnergy but with 3D cluster info 
        Note that if using maskLayer, then 3D clusters that span one layer that is exactly maskLayer are dropped.
        Index : eventInternal, clus3D_id
        Columns : intervalFractionEnergy_minLayer	intervalFractionEnergy_maxLayer intervalFractionEnergy_length beamEnergy	clus3D_x	clus3D_y	clus3D_z	clus3D_energy	clus3D_size"""
        df = self.clusters3D_intervalHoldingFractionOfEnergy(fraction=fraction, maskLayer=maskLayer).join(self.clusters3D)
        df["intervalFractionEnergy_length"] = df["intervalFractionEnergy_maxLayer"]-df["intervalFractionEnergy_minLayer"]+1
        return df


def clusters3D_filterLargestCluster(clusters3D_df : pd.DataFrame, dropDuplicates=["eventInternal"]) -> pd.Series:
    """
    df is comp.clusters3D
    Filter out, for each event, only the 3D cluster that has the highest energy
    Parameters : 
    dropDuplicates : only keep a unique value 
    """
    return clusters3D_df.reset_index().sort_values("clus3D_energy", ascending=False).drop_duplicates(dropDuplicates)


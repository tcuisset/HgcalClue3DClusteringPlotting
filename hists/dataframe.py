from functools import cached_property
import operator

import awkward as ak
import pandas as pd

def divideByBeamEnergy(df:pd.DataFrame, colName:str) -> pd.DataFrame:
    """ Add a columns named colName_fractionOfBeamEnergy """
    df[colName+"_fractionOfBeamEnergy"] = df[colName]/df["beamEnergy"]
    return df


class DataframeComputations:
    def __init__(self, tree_array:ak.Array) -> None:
        self.array = tree_array
    
    @cached_property
    def trueBeamEnergy(self) -> pd.DataFrame:
        """ If trueBeamEnergy exists, returns : 
            Columns : eventn beamEnergy, trueBeamEnergy
            Index : event
        otherwise return a dataframe with a single row : 1, 0, 0
        """
        if "trueBeamEnergy" in self.array.fields:
            return (ak.to_dataframe(self.array[
                ["beamEnergy", "trueBeamEnergy"]
                ], 
                levelname=lambda i : {0 : "event"}[i])
            )
        else:
            return pd.DataFrame({"event":[1], "beamEnergy":[0], "trueBeamEnergy":[0]}).set_index("event")

    @property
    def impact(self) -> pd.DataFrame:
        """ 
        Columns : event   layer   impactX  impactY  
        MultiIndex : (event, layer)
        """
        df = ak.to_dataframe(self.array[
            ["impactX", "impactY"]],
            levelname=lambda i : {0 : "event", 1:"layer_minus_one"}[i]).reset_index(level="layer_minus_one")
        df["layer"] = df["layer_minus_one"] + 1
        return df.drop("layer_minus_one", axis="columns").set_index("layer", append=True)
    
    @property
    def impactWithBeamEnergy(self) -> pd.DataFrame:
        """ 
        Columns : event   layer beamEnergy  impactX  impactY  
        Index : event
        """
        df = ak.to_dataframe(self.array[
            ["beamEnergy", "impactX", "impactY"]],
            levelname=lambda i : {0 : "event", 1:"layer_minus_one"}[i]).reset_index(level="layer_minus_one")
        df["layer"] = df["layer_minus_one"] + 1
        return df.drop("layer_minus_one", axis="columns")

    @cached_property
    def rechits(self) -> pd.DataFrame:
        """
        Columns : event  rechit_id  beamEnergy	rechits_x	rechits_y	rechits_z	rechits_energy	rechits_layer	rechits_rho	rechits_delta	rechits_pointType  
        MultiIndex : (event, rechit_id)
        """
        return ak.to_dataframe(self.array[
            ["beamEnergy", "rechits_x", "rechits_y", "rechits_z", "rechits_energy", "rechits_layer",
            "rechits_rho", "rechits_delta", "rechits_pointType"]], 
            levelname=lambda i : {0 : "event", 1:"rechit_id"}[i])

    @cached_property
    def rechits_totalReconstructedEnergyPerEvent(self) -> pd.DataFrame:
        """ Sum of all rechits energy per event
        Index : event
        Columns : beamEnergy rechits_energy_sum
        """
        return divideByBeamEnergy(self.rechits[["beamEnergy", "rechits_energy"]].groupby(by="event").agg(
            beamEnergy=pd.NamedAgg(column="beamEnergy", aggfunc="first"),
            rechits_energy_sum=pd.NamedAgg(column="rechits_energy", aggfunc="sum"),
        ), "rechits_energy_sum")

    @cached_property
    def layerToZMapping(self) -> dict[int, float]:
        """ Returns a dict layer_nb -> layer z position """
        return self.rechits[["rechits_z", "rechits_layer"]].groupby("rechits_layer").first()["rechits_z"].to_dict()

    @property
    def impactWithZPosition(self) -> pd.DataFrame:
        """ Transform impact df to use z layer position instead of layer number 
        MultiIndex : (event, impactZ)
        Columns : impactX, impactY
        """
        # Select only values of layer that are in the dictionnary layerToZMapping (otherwise you have a column with layer numbers and z positions at the same time)
        df = self.impact.iloc[self.impact.index.get_level_values("layer") <= max(self.layerToZMapping)].rename(
            index=self.layerToZMapping, level="layer")
        df.index.names = ["event", "impactZ"] # Rename layer -> impactZ
        return df

    @cached_property
    def clusters2D(self) -> pd.DataFrame:
        """
        Builds a pandas DataFrame holding all 2D cluster information (without any rechit info)
        MultiIndex : event  clus2D_id
        Columns :  beamEnergy	clus2D_x	clus2D_y	clus2D_z	clus2D_energy	clus2D_layer	clus2D_size	clus2D_rho	clus2D_delta	clus2D_pointType
        """
        return ak.to_dataframe(
            self.array[["beamEnergy", "clus2D_x", "clus2D_y", "clus2D_z", "clus2D_energy", "clus2D_layer", "clus2D_size",
                "clus2D_rho", "clus2D_delta", "clus2D_pointType"]],
            levelname=lambda i : {0 : "event", 1:"clus2D_id"}[i]
        )


    #Don't cache this as unlikely to be recomputed (only used by )
    @property
    def clusters2D_with_rechit_id(self) -> pd.DataFrame:
        """
        MultiIndex : event	clus2D_id	rechit_internal_id
        Columns : beamEnergy	[clus2D_x	clus2D_y	clus2D_z	clus2D_energy]	clus2D_layer	clus2D_size	clus2D_idxs	clus2D_rho	clus2D_delta	clus2D_pointType
        Note : rechit_internal_id is an identifier counting rechits in each 2D cluster (it is NOT the same as rechit_id, which is unique per event, whilst rechit_internal_id is only unique per 2D cluster)
        """
        return ak.to_dataframe(
            self.array[[
                "beamEnergy",  "clus2D_layer",
                #"clus2D_x", "clus2D_y", "clus2D_z", "clus2D_energy",
                "clus2D_rho", "clus2D_delta", "clus2D_idxs", "clus2D_pointType"
            ]], 
            levelname=lambda i : {0 : "event", 1:"clus2D_id", 2:"rechit_internal_id"}[i]
        )

    @cached_property
    def clusters2D_merged_rechit(self) -> pd.DataFrame:
        """
        MultiIndex : event	clus2D_id	rechit_internal_id
        Columns : beamEnergy	clus2D_layer	clus2D_rho	clus2D_delta	clus2D_idxs	clus2D_pointType	beamEnergy_from_rechits	rechits_x	rechits_y	rechits_z	rechits_energy	rechits_layer	rechits_rho	rechits_delta	rechits_pointType
        Note : rechit_internal_id is an identifier counting rechits in each 2D cluster (it is NOT the same as rechit_id, which is unique per event, whilst rechit_internal_id is only unique per 2D cluster)
        beamEnergy_from_rechits is just a duplicate of beamEnergy
        """
        return pd.merge(
            self.clusters2D_with_rechit_id,     # Left : clusters2D with clus2D_idxs column (one row per rechit of each 2D cluster)
            self.rechits,                       # Right : rechits
            how='inner',                        # Do an inner join (keeps only rechits that are in a 2D cluster, ie drop outliers). Left should be identical. 
            # Outer and right would include also rechits that are outliers (not associated to any cluster)
            left_on=["event", "clus2D_idxs"],   # Left ; join on columns :; event, clus2D_idxs     (where clus2D_idxs should match rechit_id)
            right_index=True,                   # Right : join on index, ie event, rechit_id
            suffixes=(None, "_from_rechits"), # Deal with columns present on both sides (currently only beamEnergy) :
            #        take the column from left with same name, the one from right renamed beamEnergy_from_rechits (they should be identical anyway)
            validate="one_to_one"               # Cross-check :  Make sure there are no weird things (such as duplicate ids), should not be needed
        )

    @cached_property
    def clusters2D_totalEnergyPerEvent(self) -> pd.DataFrame:
        """ Computer per event the total clustered energy by CLUE2D
        Index : event
        Columns : beamEnergy clus2D_energy_sum clus2D_energy_sum_fractionOfBeamEnergy
        """
        return divideByBeamEnergy(self.clusters2D[["beamEnergy", "clus2D_energy"]].groupby(by=['event']).agg(
                beamEnergy=pd.NamedAgg(column="beamEnergy", aggfunc="first"),
                clus2D_energy_sum=pd.NamedAgg(column="clus2D_energy", aggfunc="sum"),
            ), "clus2D_energy_sum")
    
    def get_clusters2D_perLayerInfo(self, withBeamEnergy=True) -> pd.DataFrame:
        """
        Compute per event and per layer the total 2D-cluster energies (and the same as a fraction of beam energy) and the number of 2D clusters
        Parameter : withBeamEnergy : whether to add beamEnergy column
        Index : event, clus2D_layer
        Column : [beamEnergy,] clus2D_energy_sum clus2D_count [clus2D_energy_sum_fractionOfBeamEnergy]
        """
        if withBeamEnergy:
            return divideByBeamEnergy(self.clusters2D[["beamEnergy", "clus2D_layer", "clus2D_energy"]].groupby(by=['event', 'clus2D_layer']).agg(
                beamEnergy=pd.NamedAgg(column="beamEnergy", aggfunc="first"),
                clus2D_energy_sum=pd.NamedAgg(column="clus2D_energy", aggfunc="sum"),
                clus2D_count=pd.NamedAgg(column="clus2D_energy", aggfunc="count")
            ), "clus2D_energy_sum")
        else:
            return self.clusters2D[["clus2D_layer", "clus2D_energy"]].groupby(by=['event', 'clus2D_layer']).agg(
                clus2D_energy_sum=pd.NamedAgg(column="clus2D_energy", aggfunc="sum"),
                clus2D_count=pd.NamedAgg(column="clus2D_energy", aggfunc="count")
            )

    @cached_property
    def clusters2D_selectLayerWithMaxClusteredEnergy(self) -> pd.DataFrame:
        """
        From comp.clusters2D, select only 2D clusters in the layer with the maximum clustered energy (per event)
        Index : event	  
        Columns : all usual clusters2D columns, including clus2D_layer
        (note that clus2D_layer is identical for all rows in a given event)
        """
        # Start from clusters2D_maxEnergyPerLayer and build the index 
        # First group by event
        # Then compute the layer nb of the maximum value clus2D_energy_sum per event
        index = self.get_clusters2D_perLayerInfo(withBeamEnergy=False)[["clus2D_energy_sum"]].groupby(by=["event"]).idxmax()
        # Index is a Series with : Index=event, Column=(event, layer) (as a tuple)

        # Reindex clusters2D so that we can apply our index
        reindexed_clus2D = self.clusters2D.reset_index().set_index(["event", "clus2D_layer"])

        # Apply the index, this will select for each event, only rows with the right layer
        return reindexed_clus2D.loc[index.clus2D_energy_sum].reset_index(level=["clus2D_layer"])

    @property
    def clusters2D_sumClustersOnLayerWithMaxClusteredEnergy(self) -> pd.DataFrame:
        """
        For each event, find the layer with the maximum clustered energy (of 2D clusters) and give the sum of 2D clustered energy on this layer (and the nb of 2D clusters)
        Columns : event clus2D_layer beamEnergy	clus2D_energy_sum	clus2D_count
        Note : resulting dataframe is not sorted by event number
        """
        # Old method : using idxmax : very slow
            # Start from clusters2D_maxEnergyPerLayer and build the index 
            # First group by event
            # Then compute the layer nb of the maximum value clus2D_energy_sum per event
            # index = self.get_clusters2D_perLayerInfo(withBeamEnergy=False)[["clus2D_energy_sum"]].groupby(by=["event"]).idxmax()
            # Index is a Series with : Index=event, Column=(event, layer) (as a tuple)

            # Apply the index, this will select for each event, only rows with the right layer
            #return self.get_clusters2D_perLayerInfo().loc[index.clus2D_energy_sum].reset_index(level=["clus2D_layer"])
        
        # New method : uses sorting and drop_duplicates (at least an order of magnitude faster, but event nb are not sorted)
        return (self.get_clusters2D_perLayerInfo()
            .reset_index() # So event is a column
            .sort_values("clus2D_energy_sum", ascending=False) # Sort descending on cluster 2D energy sum
            .drop_duplicates("event", keep="first") # Drop duplicate events, will keep only the first row ie the one with highest energy sum
        )

    @cached_property
    def clusters3D(self) -> pd.DataFrame:
        """
        MultiIndex : event   clus3D_id
        Columns :    clus3D_x	clus3D_y	clus3D_z	clus3D_energy	clus3D_layer clus3D_size
        """
        return (ak.to_dataframe(self.array[
            ["beamEnergy", "clus3D_x", "clus3D_y", "clus3D_z", "clus3D_energy", "clus3D_size"]
            ], 
            levelname=lambda i : {0 : "event", 1:"clus3D_id"}[i])
        )

    @cached_property
    def clusters3D_largestClusterIndex(self) -> pd.MultiIndex:
        """
        Compute for each event, the index of the 3D cluster with the largest clustered energy (clus3D_energy)
        (in case of equality returns the first one in dataset)
        Returns a MultiIndex (event, clus3D_id), to be used with loc (on a df indexed by (event, clus3D_id) ):
        ex : clusters3D.loc[clusters3D_largestClusterIndex]
        (not sorted)
        """
        df = (self.clusters3D
            [["clus3D_energy"]] # Dataframe is index=(event, clus3D_id), columns=clus3D_energy
            .reset_index()
            .sort_values("clus3D_energy", ascending=False)
            .drop_duplicates("event") # Keep for each event only the clus3D_id with highest clus3D_energy
            .drop(columns="clus3D_energy")
        )
        return pd.MultiIndex.from_frame(df) # Make a multiindex out of it
        # Old, slow version: 
        #return self.clusters3D.groupby(["event"])["clus3D_energy"].idxmax()

    @property
    def clusters3D_largestCluster(self) -> pd.DataFrame:
        """ 
        Same as clusters3D but only, for each event, with the 3D cluster with highest energy 
        Index : event
        Columns : beamEnergy, clus3D_* 
        """
        return self.clusters3D.loc[self.clusters3D_largestClusterIndex]

    #Don't cache this as unlikely to be needed again after caching clusters3D_merged_2D
    @property
    def clusters3D_with_clus2D_id(self) -> pd.DataFrame:
        """
        MultiIndex : event  clus3D_id  clus2D_internal_id
        Columns : 	clus3D_energy	clus3D_layer clus3D_size clus3D_idxs   # clus3D_x	clus3D_y	clus3D_z
        Note : clus2D_internal_id is an identifier counting 2D clusters in each 3D cluster (it is NOT the same as clus2D_id, which is unique per event, whilst clus2D_internal_id is only unique per 3D cluster)
        """
        return (ak.to_dataframe(
            self.array[["beamEnergy", "clus3D_energy", "clus3D_size", "clus3D_idxs"]], # "clus3D_x", "clus3D_y", "clus3D_z",
            levelname=lambda i : {0 : "event", 1:"clus3D_id", 2:"clus2D_internal_id"}[i]
        )
        .reset_index(level="clus2D_internal_id", drop=True)
        .rename(columns={"clus3D_idxs" : "clus2D_id"})
        )
        

    def clusters3D_merged_2D(self, clusters3D_with_clus2D_id_df=None) -> pd.DataFrame:
        """
        Merge the dataframe clusters3D_with_clus2D_id_df with clusters2D
        Param : clusters3D_with_clus2D_id_df : the dataframe to use for left param of join
        If None, uses self.clusters3D_with_clus2D_id
        
        Returns : 
        MultiIndex : event	clus3D_id	clus2D_internal_id
        Columns : beamEnergy	clus3D_energy	clus3D_size	clus2D_id		clus2D_x	clus2D_y	clus2D_z	clus2D_energy	clus2D_layer	clus2D_rho	clus2D_delta	clus2D_pointType
        as well as beamEnergy_from_2D_clusters which is just a duplicate of beamEnergy

        If you want only the highest energy 3D cluster you can do 
        comp.clusters3D_with_clus2D_id.pipe(clusters3D_filterLargestCluster).pipe(comp.clusters3D_merged_2D)
        """
        if clusters3D_with_clus2D_id_df is None:
            clusters3D_with_clus2D_id_df = self.clusters3D_with_clus2D_id
        return pd.merge(
            clusters3D_with_clus2D_id_df.reset_index(), # Left
            self.clusters2D,                # Right
            how='inner',                    # Inner join (the default). Keeps only 2D clusters that are associated to a 3D cluster (ie drop outliers)
            # Outer and right would include also rechits that are outliers (not associated to any cluster)
            left_on=["event", "clus2D_id"],  # Left : Join on event nb and ID of 2D cluster per event
            right_index=True,                  # Right : Join on MultiIndex, ie on event and clus2D_id
            suffixes=(None, "_from_2D_clusters"), # Deal with columns present on both sides (currently only beamEnergy) :
            #        take the column from left with same name, the one from right renamed beamEnergy_from_2D_clusters (they should be identical anyway)
            validate="one_to_one"               # Cross-check :  Make sure there are no weird things (such as duplicate ids), should not be needed
        )#.droplevel(level="clus2D_internal_id") # remove the useless clus2D_internal_id column
    

    def clusters3D_merged_2D_impact(self, clusters3D_merged_2D_df:pd.DataFrame|None = None) -> pd.DataFrame:
        """
        Merge clusters3D_merged_2D with impact dataframe, to get impact info for all 2D clusters members of a 3D cluster
        Also creates clus2D_diff_impact_x and clus2D_diff_impact_y columns holding the difference between 2D cluster position and extrapolated track impact position on layer
        """
        if clusters3D_merged_2D_df is None:
            clusters3D_merged_2D_df = self.clusters3D_merged_2D()
        merged_df = pd.merge(
            # Left : previously merged dataframe
            clusters3D_merged_2D_df,

            #Right : impact df (indexed by event and layer)
            self.impact, 

            # Map event on both sides
            # Map layer of 2D cluster with layer of impact computation
            left_on=("event", "clus2D_layer"),
            right_on=("event", "layer")
        ).set_index(["event", "clus3D_id"]) # Add clus3D_id to the index (so we can select only main 3D clusters from clusters3D_largestClusterIndex)

        merged_df["clus2D_diff_impact_x"] = merged_df["clus2D_x"] - merged_df["impactX"]
        merged_df["clus2D_diff_impact_y"] = merged_df["clus2D_y"] - merged_df["impactY"]
        return merged_df

    
    def clusters3D_firstLastLayer(self, clusters3D_merged_2D_df:pd.DataFrame) -> pd.DataFrame:
        """
        For each 3D cluster, compute the first and last layer numbers of contained 2D clusters 
        Param : clusters3D_merged_2D_df : dataframe to consider, should be a subset of self.clusters3D_merged_2D
        Returns : 
        MultiIndex : event, clus3D_id
        Columns : beamEnergy, clus3D_energy, clus2D_minLayer, clus2D_maxLayer

        Should be used as :
        self.clusters3D_with_clus2D_id[.pipe(clusters3D_filterLargestCluster)].pipe(clusters3D_merged_2D).pipe(self.clusters3D_firstLastLayer)
        """
        return (clusters3D_merged_2D_df[["event", "clus3D_id", "beamEnergy", "clus3D_energy", "clus3D_size", "clus2D_layer"]]
            .groupby(["event", "clus3D_id"])
            .agg(
                clus2D_minLayer=pd.NamedAgg(column="clus2D_layer", aggfunc="min"),
                clus2D_maxLayer=pd.NamedAgg(column="clus2D_layer", aggfunc="max"),
                beamEnergy=pd.NamedAgg(column="beamEnergy", aggfunc="first"),
                clus3D_energy=pd.NamedAgg(column="clus3D_energy", aggfunc="first"),
                clus3D_size=pd.NamedAgg(column="clus3D_size", aggfunc="first")
            )
        )

    def clusters3D_indexOf3DClustersPassingMinNumLayerCut(self, minNumLayerCluster):
        df = self.clusters3D_merged_2D[["clus2D_layer"]].groupby(["event", "clus3D_id"]).agg(
            clus2D_layer_min=pd.NamedAgg(column="clus2D_layer", aggfunc="min"),
            clus2D_layer_max=pd.NamedAgg(column="clus2D_layer", aggfunc="max"),
        )
        df["clus3D_numLayers"] = df["clus2D_layer_max"] - df["clus2D_layer_min"]+1
        return df["clus3D_numLayers"] < minNumLayerCluster
    
    @cached_property
    def clusters3D_energyClusteredPerLayer(self) -> pd.DataFrame:
        """ Compute total 2D clustered energy per layer for each 3D cluster 
        MultiIndex : event	clus3D_id	clus2D_layer
        Columns : beamEnergy clus3D_size clus3D_energy	clus2D_energy_sum
        """
        return (
            self.clusters3D_merged_2D()[["event", "clus3D_id", "beamEnergy", "clus3D_energy", "clus3D_size", "clus2D_energy", "clus2D_layer"]]

            # For each event, cluster 3D and layer, sum clus2D_energy
            .groupby(by=["event", "clus3D_id", "clus2D_layer"]).agg(
                beamEnergy=pd.NamedAgg(column="beamEnergy", aggfunc="first"),
                clus3D_size=pd.NamedAgg(column="clus3D_size", aggfunc="first"),
                clus3D_energy=pd.NamedAgg(column="clus3D_energy", aggfunc="first"),
                clus2D_energy_sum=pd.NamedAgg(column="clus2D_energy", aggfunc="sum")
            )
        )
    
    @cached_property
    def clusters3D_impact_usingLayerWithMax2DClusteredEnergy(self):
        """ Add a column to clusters3D with the difference between clus3D_x and impactX, impactX being computed on the layer with maximum 2D clustered energy (of each cluster 3D)
        Note : event is not in index and dataframe is not sorted (neither on event nor on clus3D_id)
        Columns : event clus3D_id	beamEnergy	clus3D_x	clus3D_y	clus3D_z	clus3D_energy	clus3D_size	layer_with_max_clustered_energy	impactX	impactY	clus3D_diff_impact_x clus3D_diff_impact_y
        """

        # Old way : uses idxmax, very slow
        #self.clusters3D["layer_with_max_clustered_energy"] = self.clusters3D_layerWithMaxClusteredEnergy

        # New way : uses sorting and drop_duplicates
        # First : for each event and 3D cluster find the layer with the maximum clus2D_energy_sum
        df_layerMax2DEnergy = (self.clusters3D_energyClusteredPerLayer
            .reset_index()
            .sort_values("clus2D_energy_sum", ascending=False)
            .drop_duplicates(["event", "clus3D_id"], keep="first") # Drop duplicates : will keep only highest value of clus2D_energy_sum for each event, clus3D_id
        ).rename(columns={"clus2D_layer":"layer_with_max_clustered_energy"})

        # Merge with impact
        df_withImpact = pd.merge(
            # Left
            df_layerMax2DEnergy,
            # Right :
            self.impact,

            how='left', # Left join so we preserve all 3D clusters
            # Map event and layer
            left_on=["event", "layer_with_max_clustered_energy"],
            right_on=["event", "layer"]
        )

        # We add back clus3D_x and clus3D_y by joining with clusters3D :
        final_df = pd.merge(
            df_withImpact,
            self.clusters3D[["clus3D_x", "clus3D_y"]],
            on=["event", "clus3D_id"]
        )

        # Make the substractions : 
        final_df["clus3D_diff_impact_x"] = final_df["clus3D_x"] - final_df["impactX"]
        final_df["clus3D_diff_impact_y"] = final_df["clus3D_y"] - final_df["impactY"]
        return final_df



def clusters3D_filterLargestCluster(clusters3D_df : pd.DataFrame, dropDuplicates=["event"]) -> pd.Series:
    """
    df is comp.clusters3D
    Filter out, for each event, only the 3D cluster that has the highest energy
    Parameters : 
    dropDuplicates : only keep a unique value 
    """
    return clusters3D_df.reset_index().sort_values("clus3D_energy", ascending=False).drop_duplicates(dropDuplicates)


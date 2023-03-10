from functools import cached_property

import awkward as ak
import pandas as pd

class DataframeComputations:
    def __init__(self, tree_array) -> None:
        self.array = tree_array
    
    @cached_property
    def beamEnergy(self) -> pd.DataFrame:
        return (ak.to_dataframe(self.array[
            ["beamEnergy"]
            ], 
            levelname=lambda i : {0 : "event", 1:"clus3D_id"}[i])
        )

    @property
    def impact(self) -> pd.DataFrame:
        """ 
        Columns : event   layer   impactX  impactY  
        MultiIndex : (event, layer)
        """
        return ak.to_dataframe(self.array[
            ["impactX", "impactY"]],
            levelname=lambda i : {0 : "event", 1:"layer"}[i])
    
    @property
    def impactWithBeamEnergy(self) -> pd.DataFrame:
        """ 
        Columns : event   layer beamEnergy  impactX  impactY  
        MultiIndex : (event, layer)
        """
        return ak.to_dataframe(self.array[
            ["beamEnergy", "impactX", "impactY"]],
            levelname=lambda i : {0 : "event", 1:"layer"}[i])

    @cached_property
    def rechits(self) -> pd.DataFrame:
        """
        Columns : event  rechit_id  beamEnergy	rechits_x	rechits_y	rechits_z	rechits_energy	rechits_layer	rechits_rho	rechits_delta	rechits_isSeed  
        MultiIndex : (event, rechit_id)
        """
        return ak.to_dataframe(self.array[
            ["beamEnergy", "rechits_x", "rechits_y", "rechits_z", "rechits_energy", "rechits_layer",
            "rechits_rho", "rechits_delta", "rechits_isSeed"]], 
            levelname=lambda i : {0 : "event", 1:"rechit_id"}[i])

    @cached_property
    def clusters2D(self) -> pd.DataFrame:
        """
        Builds a pandas DataFrame holding all 2D cluster information (without any rechit info)
        MultiIndex : event  clus2D_id
        Columns :  beamEnergy	clus2D_x	clus2D_y	clus2D_z	clus2D_energy	clus2D_layer	clus2D_size	clus2D_rho	clus2D_delta	clus2D_isSeed
        """
        return ak.to_dataframe(
            self.array[["beamEnergy", "clus2D_x", "clus2D_y", "clus2D_z", "clus2D_energy", "clus2D_layer",
                "clus2D_rho", "clus2D_delta", "clus2D_isSeed"]],
            levelname=lambda i : {0 : "event", 1:"clus2D_id"}[i]
        )


    #Don't cache this as unlikely to be recomputed (only used by )
    @property
    def clusters2D_with_rechit_id(self) -> pd.DataFrame:
        """
        MultiIndex : event	clus2D_id	rechit_internal_id
        Columns : beamEnergy	[clus2D_x	clus2D_y	clus2D_z	clus2D_energy]	clus2D_layer	clus2D_size	clus2D_idxs	clus2D_rho	clus2D_delta	clus2D_isSeed
        Note : rechit_internal_id is an identifier counting rechits in each 2D cluster (it is NOT the same as rechit_id, which is unique per event, whilst rechit_internal_id is only unique per 2D cluster)
        """
        return ak.to_dataframe(
            self.array[[
                "beamEnergy",  "clus2D_layer",
                #"clus2D_x", "clus2D_y", "clus2D_z", "clus2D_energy",
                "clus2D_rho", "clus2D_delta", "clus2D_idxs", "clus2D_isSeed"
            ]], 
            levelname=lambda i : {0 : "event", 1:"clus2D_id", 2:"rechit_internal_id"}[i]
        )

    @cached_property
    def clusters2D_merged_rechit(self) -> pd.DataFrame:
        """
        MultiIndex : event	clus2D_id	rechit_internal_id
        Columns : beamEnergy	clus2D_layer	clus2D_rho	clus2D_delta	clus2D_idxs	clus2D_isSeed	beamEnergy_from_rechits	rechits_x	rechits_y	rechits_z	rechits_energy	rechits_layer	rechits_rho	rechits_delta	rechits_isSeed
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

    
    def get_clusters2D_perLayerInfo(self, withBeamEnergy=True) -> pd.DataFrame:
        """
        Compute per event and per layer the total 2D-cluster energies and the number of 2D clusters
        Parameter : withBeamEnergy : whether to add beamEnergy column
        Index : event, clus2D_layer
        Column : [beamEnergy,] clus2D_energy_sum clus2D_count
        """
        if withBeamEnergy:
            return self.clusters2D[["beamEnergy", "clus2D_layer", "clus2D_energy"]].groupby(by=['event', 'clus2D_layer']).agg(
                beamEnergy=pd.NamedAgg(column="beamEnergy", aggfunc="first"),
                clus2D_energy_sum=pd.NamedAgg(column="clus2D_energy", aggfunc="sum"),
                clus2D_count=pd.NamedAgg(column="clus2D_energy", aggfunc="count")
            )
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
        Index : event
        Columns : clus2D_layer beamEnergy	clus2D_energy_sum	clus2D_count
        """
        # Start from clusters2D_maxEnergyPerLayer and build the index 
        # First group by event
        # Then compute the layer nb of the maximum value clus2D_energy_sum per event
        index = self.get_clusters2D_perLayerInfo(withBeamEnergy=False)[["clus2D_energy_sum"]].groupby(by=["event"]).idxmax()
        # Index is a Series with : Index=event, Column=(event, layer) (as a tuple)

        # Apply the index, this will select for each event, only rows with the right layer
        return self.get_clusters2D_perLayerInfo().loc[index.clus2D_energy_sum].reset_index(level=["clus2D_layer"])


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
    def clusters3D_largestClusterIndex(self) -> pd.Series:
        """
        Compute for each event, the index of the 3D cluster with the largest clustered energy (clus3D_energy)
        (in case of equality returns the first one in dataset)
        Returns a series of tuples (event, clus3D_id), to be used with loc (on a df indexed by (event, clus3D_id) ):
        ex : clusters3D.loc[clusters3D_largestClusterIndex]
        """
        return self.clusters3D.groupby(["event"])["clus3D_energy"].idxmax()

    @property
    def clusters3D_largestCluster(self) -> pd.DataFrame:
        """ 
        Same as clusters3D but only, for each event, with the 3D cluster with highest energy 
        Index : event
        Columns : beamEnergy, clus3D_* 
        """
        return self.clusters3D.loc[self.clusters3D_largestClusterIndex].reset_index(level="clus3D_id")

    #Don't cache this as unlikely to be needed again after caching clusters3D_merged_2D
    @property
    def clusters3D_with_clus2D_id(self) -> pd.DataFrame:
        """
        MultiIndex : event  clus3D_id  clus2D_internal_id
        Columns : 	clus3D_energy	clus3D_layer clus3D_size clus3D_idxs   # clus3D_x	clus3D_y	clus3D_z
        Note : clus2D_internal_id is an identifier counting 2D clusters in each 3D cluster (it is NOT the same as clus2D_id, which is unique per event, whilst clus2D_internal_id is only unique per 3D cluster)
        """
        return ak.to_dataframe(
            self.array[["beamEnergy", "clus3D_energy", "clus3D_size", "clus3D_idxs"]], # "clus3D_x", "clus3D_y", "clus3D_z",
            levelname=lambda i : {0 : "event", 1:"clus3D_id", 2:"clus2D_internal_id"}[i]
        )

    @cached_property
    def clusters3D_merged_2D(self) -> pd.DataFrame:
        """
        Merge the dataframe clusters3D with clusters2D

        MultiIndex : event	clus3D_id	clus2D_internal_id
        Columns : beamEnergy	clus3D_energy	clus3D_size	clus3D_idxs		clus2D_x	clus2D_y	clus2D_z	clus2D_energy	clus2D_layer	clus2D_rho	clus2D_delta	clus2D_isSeed
        as well as beamEnergy_from_2D_clusters which is just a duplicate of beamEnergy
        Note : clus2D_id gets lost in the pd.merge due to it being in a MultiIndex, use clus3D_idxs  which is the same

        If you want only the highest energy 3D cluster you can do 
        clusters3D_merged_2D.loc[comp.clusters3D_largestClusterIndex]
        """
        return pd.merge(
            self.clusters3D_with_clus2D_id, # Left
            self.clusters2D,                # Right
            how='inner',                    # Inner join (the default). Keeps only 2D clusters that are associated to a 3D cluster (ie drop outliers)
            # Outer and right would include also rechits that are outliers (not associated to any cluster)
            left_on=["event", "clus3D_idxs"],  # Left : Join on event nb and ID of 2D cluster per event
            right_index=True,                  # Right : Join on MultiIndex, ie on event and clus2D_id
            suffixes=(None, "_from_2D_clusters"), # Deal with columns present on both sides (currently only beamEnergy) :
            #        take the column from left with same name, the one from right renamed beamEnergy_from_2D_clusters (they should be identical anyway)
            validate="one_to_one"               # Cross-check :  Make sure there are no weird things (such as duplicate ids), should not be needed
        ).droplevel(level="clus2D_internal_id") # remove the useless clus2D_internal_id column
    
    @cached_property
    def clusters3D_merged_2D_impact(self) -> pd.DataFrame:
        """
        TODO this join is probably not needed, faster to just lookup in impact df for all rows of clusters3D_merged_2D
        Merge clusters3D_merged_2D with impact dataframe, to get impact info for all 2D clusters members of a 3D cluster
        Also creates clus2D_diff_impact_x and clus2D_diff_impact_y columns holding the difference between 2D cluster position and extrapolated track impact position on layer
        """
        merged_df = pd.merge(
            # Left : previously merged dataframe
            self.clusters3D_merged_2D.reset_index(level="clus3D_id"), # reset index clus3D_id otherwise it gets lost during the join

            #Right : impact df (indexed by event and layer)
            self.impact, 

            # Map event on both sides
            # Map layer of 2D cluster with layer of impact computation
            left_on=("event", "clus2D_layer"),
            right_on=("event", "layer")
        ).set_index("clus3D_id", append=True) # Add clus3D_id to the index (so we can select only main 3D clusters from clusters3D_largestClusterIndex)

        merged_df["clus2D_diff_impact_x"] = merged_df["clus2D_x"] - merged_df["impactX"]
        merged_df["clus2D_diff_impact_y"] = merged_df["clus2D_y"] - merged_df["impactY"]
        return merged_df

    @cached_property
    def clusters3D_firstLastLayer(self) -> pd.DataFrame:
        """
        For each 3D cluster, compute the first and last layer numbers of contained 2D clusters 
        MultiIndex : event, clus3D_id
        Columns : beamEnergy, clus3D_energy, clus2D_minLayer, clus2D_maxLayer
        """
        return (self.clusters3D_merged_2D[["beamEnergy", "clus3D_energy", "clus3D_size", "clus2D_layer"]]
            .groupby(level=["event", "clus3D_id"])
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
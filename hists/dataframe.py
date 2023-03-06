from functools import cached_property

import awkward as ak
import pandas as pd

class DataframeComputations:
    def __init__(self, tree_array) -> None:
        self.array = tree_array
    
    @cached_property
    def impact(self) -> pd.DataFrame:
        """ 
        Columns : event   layer   impactX  impactY  
        MultiIndex : (event, layer)
        """
        return ak.to_dataframe(self.array[
            ["impactX", "impactY"]],
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


    #Don't cache this as unlikely to be needed again after caching clusters3D_merged_2D
    @property
    def clusters3D_with_clus2D_id(self) -> pd.DataFrame:
        """
        MultiIndex : event  clus3D_id  clus2D_internal_id
        Columns : 	clus3D_energy	clus3D_layer clus3D_size clus3D_idxs   # clus3D_x	clus3D_y	clus3D_z
        """
        return ak.to_dataframe(
            self.array[["beamEnergy", "clus3D_energy", "clus3D_size", "clus3D_idxs"]], # "clus3D_x", "clus3D_y", "clus3D_z",
            levelname=lambda i : {0 : "event", 1:"clus3D_id", 2:"clus2D_internal_id"}[i]
        )

    @cached_property
    def clusters3D_merged_2D(self) -> pd.DataFrame:
        """
        Note : clus2D_internal_id is an identifier counting 2D clusters in each 3D cluster (it is NOT the same as clus2D_id, which is unique per event, whilst clus2D_internal_id is only unique per 3D cluster)
        clus2D_id gets lost in the pd.merge due to it being in a MultiIndex, should call reset_index and the, use right_on= to keep it if you need it 
        MultiIndex : event	clus3D_id	clus2D_internal_id
        Columns : beamEnergy	clus3D_energy	clus3D_size	clus3D_idxs		clus2D_x	clus2D_y	clus2D_z	clus2D_energy	clus2D_layer	clus2D_rho	clus2D_delta	clus2D_isSeed
        as well as beamEnergy_from_2D_clusters which is just a duplicate of beamEnergy
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
        )
    
    @cached_property
    def clusters3D_merged_2D_impact(self) -> pd.DataFrame:
        merged_df = pd.merge(
            # Left : previously merged dataframe
            self.clusters3D_merged_2D,

            #Right : impact df (indexed by event and layer)
            self.impact, 

            # Map event on both sides
            # Map layer of 2D cluster with layer of impact computation
            left_on=("event", "clus2D_layer"),
            right_on=("event", "layer")
        )
        merged_df["clus2D_diff_impact_x"] = merged_df["clus2D_x"] - merged_df["impactX"]
        merged_df["clus2D_diff_impact_y"] = merged_df["clus2D_y"] - merged_df["impactY"]
        return merged_df


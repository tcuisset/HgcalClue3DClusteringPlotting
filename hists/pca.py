import numpy as np
import pandas as pd
from wpca import WPCA


def Cluster3D_PCA(df:pd.DataFrame):
    """ Do weighted PCA on a 3D cluster
    Parameter : df : dataframe holding all 2D clusters of a *single* 3D cluster, with at least clus2D_x, y, z, and clus2D_energy columns
    Returns first principal component as numpy array, shape=(3)
    """
    #if df.clus3D_size.iloc[0] <= 1:
    #    return np.array([], dtype=np.float32)
    array = df[["clus2D_x", "clus2D_y", "clus2D_z"]].to_numpy()
    weights = df[["clus2D_energy", "clus2D_energy", "clus2D_energy"]].to_numpy()
    pca = WPCA(n_components=3)
    try:
        pca.fit(array, weights=weights)
    except FloatingPointError:
        return np.array([], dtype=np.float32)
    return pca.components_[0]
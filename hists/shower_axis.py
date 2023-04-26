import numpy as np
import pandas as pd

from .pca import Cluster3D_PCA
from .parameters import layerToZMapping

def angleBetweenVectors(u, v):
    """ Compute angle between PCA and impact vectors dealing with edge cases https://stackoverflow.com/a/13849249 
    adapted to use abs so angle is always between 0 and pi/2 """ 
    # using abs 
    return np.arccos(np.clip(np.abs(np.dot(u, v)), 0, 1.0))
    #return np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))
    

def projectVectorOn(u:np.ndarray, axis:str):
    """ Project vector u on plane : 
     - axis="x" -> project on (Oxz)
     - axis="y" -> project on (Oyz)
    Also ensures vector has positive z component """
    u = np.copy(u)
    if axis == "x":
        u[1] = 0 # set y to 0
    elif axis == "y":
        u[0] = 0 # set x to 0
    else:
        raise ValueError()
    if u[2] < 0: # u.z < 0 : flip u around (for PCA)
        return -1*u
    else:
        return u

def angleInPlane(u, v, axis:str) -> float:
    """ Angle (signed) between u and v vectors, using axis :
        - "x" : angle in (Oxz) plane
        - "y" : angle in (Oyz) plane
    Taken from https://stackoverflow.com/a/33920320
    """
    u = projectVectorOn(u, axis)
    v = projectVectorOn(v, axis)
    if axis == "x":
        axis_vector = np.array([0., 1., 0.]) # y axis
    elif axis == "y":
        axis_vector = np.array([1., 0., 0.]) # x axis
    else:
        raise ValueError()
    return np.arctan2(np.dot(np.cross(u, v), axis_vector), np.dot(u, v))



def computeAngleSeries(df:pd.DataFrame, name):
    """ Parameters : 
    - name to insert 
    Returns a Series with index : value : 
    - clus3D_pca_impact_NAME_angle : the angle (in [0, pi/2] range) between PCA estimate and DWC track
    - clus3D_pca_impact_NAME_angle_x : the angle of the vectors projected in (Oxz) plane (in [-pi/2, pi/2] range, right-handed, angle PCA-> impact)
    - clus3D_pca_impact_NAME_angle_y : same but in (Oyz) plane
    """
    pca_axis = Cluster3D_PCA(df)

    # Compute DWC impact vector, using the impact point on the last and first layer of 3D cluster
    impact_df = (df
        #[["clus2D_layer", "impactX", "impactY"]]
        .sort_values("clus2D_layer")
    )
    impact_dx = impact_df.impactX.iloc[-1] - impact_df.impactX.iloc[1]
    impact_dy = impact_df.impactY.iloc[-1] - impact_df.impactY.iloc[1]
    # Get z position of last and first layer
    impact_dz = layerToZMapping[impact_df.clus2D_layer.iloc[-1]] - layerToZMapping[impact_df.clus2D_layer.iloc[1]]
    impact_vector = np.array([impact_dx, impact_dy, impact_dz])
    impact_vector = impact_vector / np.linalg.norm(impact_vector)
    #return pca_axis, impact_vector
    angle = angleBetweenVectors(pca_axis, impact_vector)
    angle_x = angleInPlane(pca_axis, impact_vector, "x")
    angle_y = angleInPlane(pca_axis, impact_vector, "y")

    prefix = "clus3D_pca_impact_" + name + "_angle"
    return pd.Series(data=[angle, angle_x, angle_y], index=[prefix, prefix+"_x", prefix+"_y"])
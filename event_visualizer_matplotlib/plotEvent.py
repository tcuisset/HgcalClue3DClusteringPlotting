import matplotlib.pyplot as plt
import uproot
import awkward as ak

def plot(tree, ax):
    pass

def plot_scatter(ax:plt.Axes, evt_array, coord1, coord2, rechits=True, clus2D=True, clus3D=True):
    if rechits:
        ax.scatter(evt_array["clus2D_"+coord1],evt_array["clus2D_"+coord2],s=evt_array["clus2D_energy"]*10,alpha=0.4,c='r',label="2D cluster")
    if clus2D:
        ax.scatter(evt_array["clus3D_"+coord1],evt_array["clus3D_"+coord2],s=evt_array["clus3D_energy"]*10,alpha=0.4,c='g',label="3D cluster")
    if clus3D:
        ax.scatter(evt_array["rechits_"+coord1],evt_array["rechits_"+coord2],s=evt_array["rechits_energy"]*10,alpha=0.4,c='b',label="rechit")
    ax.set_xlabel(coord1 + ' (cm)')
    ax.set_ylabel(coord2 + ' (cm)')

def plot_scatter_single_layer(ax:plt.Axes, evt_array, layer):
    pass

def plot_arrow_2dclusters(ax:plt.Axes, evt_array, coord1, coord2):
    """
    evt is the event number
    coord1, coord2 are the axis names of the plot can be x, y or z
    """
    for (cluster_id, hits_ids) in enumerate(evt_array["clus2D_idxs"]):
        for hit_id in hits_ids:
            ax.arrow(evt_array["rechits_"+coord1][hit_id], evt_array["rechits_"+coord2][hit_id],
                evt_array["clus2D_"+coord1][cluster_id] - evt_array["rechits_"+coord1][hit_id], 
                evt_array["clus2D_"+coord2][cluster_id] - evt_array["rechits_"+coord2][hit_id],
                width=0.05, linewidth=0.1)
            
def plot_arrow_3dclusters(ax:plt.Axes, evt_array, coord1, coord2):
    """
    Plot arrows linking 2D clusters to their corresponding 2D cluster
    evt is the event number
    coord1, coord2 are the axis names of the plot can be x, y or z
    """
    for (cluster3d_id, clusters2D_ids) in enumerate(evt_array["clus3D_idxs"]):
        for cluster2d_id in clusters2D_ids:
            ax.arrow(evt_array["clus2D_"+coord1][cluster2d_id], evt_array["clus2D_"+coord2][cluster2d_id],
                evt_array["clus3D_"+coord1][cluster3d_id] - evt_array["clus2D_"+coord1][cluster2d_id], 
                evt_array["clus3D_"+coord2][cluster3d_id] - evt_array["clus2D_"+coord2][cluster2d_id],
                width=0.05, linewidth=0.1, color='m')



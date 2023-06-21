""" Implementation of DynamicReductionNetwork, taken from Shamik 
see https://arxiv.org/abs/2003.08013 (A Dynamic Reduction Network for Point Clouds)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch.utils.checkpoint import checkpoint
from torch_cluster import knn_graph

from torch_geometric.nn import EdgeConv, NNConv
#from torch_geometric.nn.pool.edge_pool import EdgePooling

from torch_geometric.utils import normalized_cut
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.nn import (graclus, max_pool, max_pool_x,
                                global_mean_pool, global_max_pool,
                                global_add_pool,BatchNorm)

transform = T.Cartesian(cat=False)

def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))

class DynamicReductionNetwork(nn.Module):
    """ This model iteratively contracts nearest neighbour graphs until there is one output node.
    The latent space trained to group useful features at each level of aggregration.
    This allows single quantities to be regressed from complex point counts
    in a location and orientation invariant way.
    One encoding layer is used to abstract away the input features. 
    """

    def __init__(self, input_dim:int=5, hidden_dim:int=64, output_dim:int=1, k:int=16, aggr:str='add',
                 norm:torch.Tensor|list[float]=torch.tensor([1./500., 1./500., 1./54., 1/25., 1./1000.]),
                dropout:float=0.2):
 #                norm=torch.tensor([1., 1., 1., 1., 1.])):
        super(DynamicReductionNetwork, self).__init__()
        if isinstance(norm, list):
            norm = torch.tensor(norm)
        self.hyperparameters = {"drn.hidden_dim":hidden_dim, "drn.k":k, "drn.aggr":aggr, "drn.dropout":dropout}
        self.hyperparameters.update({f"drn.norm_{i}":norm_val for i, norm_val in enumerate(norm.numpy())})
        self.datanorm = nn.Parameter(norm,requires_grad=True)
        
        self.k = k
        start_width = 2 * hidden_dim
        middle_width = 3 * hidden_dim // 2

        
        
        self.inputnet =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim*2),            
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.ELU(),
            nn.Dropout(dropout),
#            nn.Linear(hidden_dim*2, hidden_dim*2),
#            nn.ELU(),
#            nn.Dropout(dropout),
#            nn.Linear(hidden_dim*2, hidden_dim*2),
#            nn.ELU(),
#            nn.Dropout(dropout),
#            nn.Linear(hidden_dim*2, hidden_dim*2),
#            nn.ELU(),
#            nn.Linear(hidden_dim*2, hidden_dim*2),
#            nn.ELU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ELU(),
        )
        
        
        convnn1 = nn.Sequential(nn.Linear(start_width, middle_width),
                                nn.ELU(),
                                nn.Dropout(dropout),
                                nn.Linear(middle_width, hidden_dim),                                             
                                nn.ELU()
                                )
        convnn2 = nn.Sequential(nn.Linear(start_width, middle_width),
                                nn.ELU(),
                                nn.Dropout(dropout),
                                nn.Linear(middle_width, hidden_dim),                                             
                                nn.ELU()
                                )
        
        convnn3 = nn.Sequential(nn.Linear(start_width, middle_width),
                                nn.ELU(),
                                nn.Dropout(dropout),
                                nn.Linear(middle_width, hidden_dim),                                             
                                nn.ELU()
                                )
                
        self.edgeconv1 = EdgeConv(nn=convnn1, aggr=aggr)
        self.edgeconv2 = EdgeConv(nn=convnn2, aggr=aggr)
        self.edgeconv3 = EdgeConv(nn=convnn3, aggr=aggr)
        
        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                    nn.ELU(),
                                    nn.Dropout(dropout),
                                    #nn.Softplus(),
                                    nn.Linear(hidden_dim, hidden_dim//2),
                                    nn.ELU(),
                                    nn.Dropout(dropout),
                                    #nn.Softplus(),
                                    nn.Linear(hidden_dim//2, hidden_dim//2),#added
                                    nn.ELU(),
                                    nn.Dropout(dropout),
                                    #nn.Softplus(),
                                    nn.Linear(hidden_dim//2, output_dim)
                                   )
        self.batchnorm1 = BatchNorm(hidden_dim)
        
    def forward(self, data):
        """ Uses only data.x and data.batch """
        data.x = self.datanorm * data.x
        data.x = self.inputnet(data.x)
        
        #print(data.batch)
#        data.x = self.batchnorm1(data.x)
        data.edge_index = to_undirected(knn_graph(data.x, self.k, data.batch, loop=False, flow=self.edgeconv1.flow))
        data.x = self.edgeconv1(data.x, data.edge_index)
        
        weight = normalized_cut_2d(data.edge_index, data.x)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data)
        
        ####
#        data.x = self.batchnorm1(data.x)
        data.edge_index = to_undirected(knn_graph(data.x, self.k, data.batch, loop=False, flow=self.edgeconv3.flow))
        data.x = self.edgeconv3(data.x, data.edge_index)
        
        weight = normalized_cut_2d(data.edge_index, data.x)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data)
        ####
        ####
        #data.edge_index = to_undirected(knn_graph(data.x, self.k, data.batch, loop=False, flow=self.edgeconv1.flow))
        #data.x = self.edgeconv3(data.x, data.edge_index)
        
        #weight = normalized_cut_2d(data.edge_index, data.x)
        #cluster = graclus(data.edge_index, weight, data.x.size(0))
        #data.edge_attr = None
        #data = max_pool(cluster, data)
        ####
#        data.x = self.batchnorm1(data.x)
        data.edge_index = to_undirected(knn_graph(data.x, self.k, data.batch, loop=False, flow=self.edgeconv2.flow))
        data.x = self.edgeconv2(data.x, data.edge_index)
        
        weight = normalized_cut_2d(data.edge_index, data.x)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        x, batch = max_pool_x(cluster, data.x, data.batch)

        x = global_max_pool(x, batch)
#        print(self.output(x))
        return self.output(x).squeeze(-1)
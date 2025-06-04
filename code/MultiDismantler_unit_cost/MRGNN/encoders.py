import sys

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F



class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self,feature_dim, 
            embed_dim,device,
            num_sample=5,
            base_model=None, gcn=False,cuda=False,
            feature_transform=False):
        super(Encoder, self).__init__()
        self.num_sample = num_sample
        self.feat_dim = feature_dim
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.device = device
        self.rand_generator = lambda mean, std, size: torch.fmod(torch.normal(mean, std, size=size), 2)
        self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        init.xavier_uniform_(self.weight)
#        self.nodeLayWeight = NodeLayWeightCos(feat_embedd)
    def forward(self, nodes, features, adj_lists, aggregator):
        #self.fea_.append(torch.matmul(self.feat_embedd,self.trans_.cpu()))
        #print(nodes)
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        #original
        aggregator.cuda = self.cuda
        neigh_feats = aggregator.forward(features, nodes, [adj_lists[int(node)] for node in nodes],self.num_sample)
        #temp
        #neigh_feats = self.aggregator.forward(torch.LongTensor(self.features(nodes)), tourch.LongTensor(self.features([self.adj_lists[int(node)] for node in nodes])))
        if not self.gcn:
            # if self.cuda:
            #     self_feats = features(torch.LongTensor(nodes).cuda(self.device))
            # else:
            #     self_feats = features(torch.LongTensor(nodes))
            combined = torch.cat([features, neigh_feats], dim=1)
        else:
            combined = neigh_feats
        combined = F.relu(torch.matmul(self.weight,combined.t()))
#        if type(nodes)==torch.Tensor:
#            nodes = nodes.tolist()
#        self.fea_[0][nodes]=combined.t().clone().cpu()
        return combined

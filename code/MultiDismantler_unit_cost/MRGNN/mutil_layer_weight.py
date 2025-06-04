import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

'''
层间节点权重计算
'''
class LayerNodeAttention_weight(nn.Module):
    def __init__(self, features_num, dropout, alpha, metapath_number ,layer_predict=0):
        super(LayerNodeAttention_weight, self).__init__()
        self.features_num = features_num
        self.layer_predict = layer_predict
        self.metapath_number = metapath_number
        self.dropout = dropout
        self.alpha = alpha
        self.leakyReLU = nn.LeakyReLU(alpha)
        self.trans = nn.Parameter(torch.eye(features_num))
        self.attention = nn.Parameter(torch.empty(1, 2 * features_num))
        nn.init.xavier_uniform_(self.attention.data, 1.414)
        self.bias = nn.Parameter(torch.Tensor(features_num))
        nn.init.zeros_(self.bias)
        self.tanh = nn.Tanh()



    def forward(self, node_features, nodes_ori):
        nodes_ori = np.array(nodes_ori)
        nodes_ori = nodes_ori.tolist()
        nodes = torch.tensor(np.unique(nodes_ori))
        temp_fea_t = []
        for kk in range(self.metapath_number):
            temp_fea_t.append(torch.zeros(len(nodes),node_features[kk].shape[1]))
        for i in range(len(nodes)):
            index_ = nodes_ori.index(nodes[i])
            for kj in range(self.metapath_number):
                temp_fea_t[kj][i] = node_features[kj][index_]
        for kl in range(self.metapath_number):
            node_features[kl] = torch.tanh(torch.matmul(temp_fea_t[kl],self.trans)+self.bias)
        node_features = torch.stack(node_features)
        layer_all_attention = torch.stack([
            self.layer_node_attention(node_features, i) for i in range(node_features.shape[1])
        ])
        Z = torch.zeros(node_features.shape[1],node_features.shape[2])
        for i in range(node_features.shape[1]):
            adj = layer_all_attention[i] 
            weight = [0 for i in range(self.metapath_number)]
            for j in range(adj.shape[0]):
                if j == self.layer_predict:
                    continue
                cat_hi = torch.cat((adj[self.layer_predict],adj[j]), dim=0)
                weight_t = math.exp(self.leakyReLU(self.attention.matmul(cat_hi)))
                weight[j] = weight_t if weight_t<1 else 1
            temp = Z[i]
            for k in range(adj.shape[0]):
                if k==self.layer_predict:
                    continue
                temp +=(weight[k] / sum(weight)) *adj[k]
            Z[i] = temp
        X = node_features[self.layer_predict]+Z
        result = torch.zeros(len(nodes_ori),X.shape[1])
        nodes_tolist = nodes.tolist()
        for m in range(len(nodes_ori)):
            index_nodes = nodes_tolist.index(nodes_ori[m])
            result[m] = X[index_nodes]
        return  result

    def transZshape(self, z, dim, i):
        matrics = torch.zeros(dim, 1)
        matrics[i, 0] = 1
        m = z.matmul(matrics).reshape(z.shape[0], z.shape[1])
        return m 

    def layer_node_attention(self, node_features, i):
        a_temp = torch.zeros(node_features.shape[1], 1)  
        a_temp[i, 0] = 1
        layer_attention = torch.transpose(node_features, 2, 1)
        b = layer_attention.matmul(a_temp).reshape(layer_attention.shape[0], layer_attention.shape[1])
        return b

class Cosine_similarity(nn.Module):
    def __init__(self, features_num, dropout, alpha, metapath_number, layer_predict=0):
        super(Cosine_similarity, self).__init__()
        self.features_num = features_num
        self.layer_predict = layer_predict
        self.metapath_number = metapath_number
        self.dropout = dropout
        self.alpha = alpha
        self.leakyReLU = nn.LeakyReLU(alpha) 
        self.trans = nn.Parameter(torch.eye(features_num))
        self.attention = nn.Parameter(torch.empty(1, 2 * features_num))
        nn.init.xavier_uniform_(self.attention.data, 1.414)
        self.bias = nn.Parameter(torch.Tensor(features_num))
        nn.init.zeros_(self.bias)

    def forward(self, node_features1, node_features2, node_features3, nodes_ori):
        nodes_ori = np.array(nodes_ori)
        nodes_ori = nodes_ori.tolist()
        nodes = torch.tensor(np.unique(nodes_ori))
        temp_fea_1 = torch.zeros(len(nodes), node_features1.shape[1])
        temp_fea_2 = torch.zeros(len(nodes), node_features2.shape[1])
        temp_fea_3 = torch.zeros(len(nodes), node_features3.shape[1])
        for i in range(len(nodes)):
            index_ = nodes_ori.index(nodes[i])
            temp_fea_1[i] = node_features1[index_]
            temp_fea_2[i] = node_features2[index_]
            temp_fea_3[i] = node_features3[index_]
        node_features1 = temp_fea_1
        node_features2 = temp_fea_2
        node_features3 = temp_fea_3

        node_features1 = torch.tanh(torch.matmul(node_features1,self.trans)+self.bias)
        node_features2 = torch.tanh(torch.matmul(node_features2,self.trans)+self.bias)
        node_features3 = torch.tanh(torch.matmul(node_features3,self.trans)+self.bias)

        node_features = torch.stack([node_features1, node_features2, node_features3])  # 层间特征矩阵拼接，变成三维

        layer_all_attention = torch.stack([
            self.layer_node_attention(node_features, i) for i in range(node_features.shape[1])
        ])

        Z = torch.zeros(node_features.shape[1],node_features.shape[2])
        for i in range(layer_all_attention.shape[0]): 
            adj = layer_all_attention[i] 
            weight = [0 for i in range(self.metapath_number)]
            for j in range(adj.shape[0]):
                if j == self.layer_predict:
                    continue
                weight[j] = F.cosine_similarity(adj[self.layer_predict], adj[j], dim=0)
            temp = Z[i]
            for k in range(adj.shape[0]):
                if k == self.layer_predict:
                    continue
                temp = temp + (weight[k] / sum(weight)) * adj[k]
            Z[i] = temp


        X = node_features[self.layer_predict]+Z
        result = torch.zeros(len(nodes_ori), X.shape[1])
        nodes_tolist = nodes.tolist()
        for m in range(len(nodes_ori)):
            index_nodes = nodes_tolist.index(nodes_ori[m])
            result[m] = X[index_nodes]
        return result


    def layer_node_attention(self, node_features, i):
        a_temp = torch.zeros(node_features.shape[1], 1) 
        a_temp[i, 0] = 1
        layer_attention = torch.transpose(node_features, 2, 1)
        b = layer_attention.matmul(a_temp).reshape(layer_attention.shape[0], layer_attention.shape[1])
        return b

class SemanticAttention(nn.Module):
    def __init__(self, features_num, dropout, alpha, metapath_number ,layer_predict=0):
        super(SemanticAttention, self).__init__()
        self.features_num = features_num
        self.layer_predict = layer_predict
        self.metapath_number = metapath_number
        self.dropout = dropout
        self.alpha = alpha
        self.leakyReLU = nn.LeakyReLU(alpha) 
        self.trans = nn.Parameter(torch.eye(features_num))
        self.attention = nn.Parameter(torch.empty(1, 2 * features_num))
        nn.init.xavier_uniform_(self.attention.data, 1.414)
        self.W = nn.Parameter(torch.empty(features_num, features_num))
        nn.init.xavier_uniform_(self.W.data, 1.414)
        self.b = nn.Parameter(torch.empty(1, features_num))
        nn.init.xavier_uniform_(self.b.data, 1.414)
        self.q = nn.Parameter(torch.empty(features_num, 1))
        nn.init.xavier_uniform_(self.q.data, 1.414)
        self.bias = nn.Parameter(torch.Tensor(features_num))
        nn.init.zeros_(self.bias)
        self.tanh = nn.Tanh()


    def forward(self, node_features, nodes_ori):
        nodes_ori = np.array(nodes_ori)
        nodes_ori = nodes_ori.tolist()
        nodes = torch.tensor(np.unique(nodes_ori))
        temp_fea_t = []
        for kk in range(self.metapath_number):
            temp_fea_t.append(torch.zeros(len(nodes),node_features[kk].shape[1]))
        for i in range(len(nodes)):
            index_ = nodes_ori.index(nodes[i])
            for kj in range(self.metapath_number):
                temp_fea_t[kj][i] = node_features[kj][index_]
        for kl in range(self.metapath_number):
            node_features[kl] = torch.tanh(torch.matmul(temp_fea_t[kl],self.trans)+self.bias)
 
        node_features = torch.stack(node_features)

        layer_all_attention = torch.stack([
            self.layer_node_attention(node_features, i) for i in range(node_features.shape[1])
        ])
        semantic_pernode = self.layer_semantic(layer_all_attention)
        Z = torch.zeros(node_features.shape[1],node_features.shape[2])
        all_weight = []
        for i in range(semantic_pernode.shape[0]): 
            adj_node = layer_all_attention[i]
            adj = semantic_pernode[i]
            trans = torch.tanh(adj.matmul(self.W)+self.b)
            w_meta = trans.matmul(self.q).reshape(trans.shape[0],trans.shape[1])#Tensor(meta_path-1,2)
            w_meta = w_meta.sum(dim=1) / w_meta.shape[1]
            beta = F.softmax(w_meta,dim=-1)
            weight = []
            for kk, weight_k in enumerate(beta):
                weight.append(weight_k)
            temp_adj = Z[i]
            index = 0
            for k_ in range(adj_node.shape[0]):
                if k_ == self.layer_predict:
                    continue
                temp_adj = temp_adj + (weight[index]/(sum(weight))) * adj_node[k_]
                index = index+1
            Z[i] = temp_adj
            all_weight.append([vitem.item() for vitem in weight])
            
        X = node_features[self.layer_predict]+Z
        result = torch.zeros(len(nodes_ori),X.shape[1])
        nodes_tolist = nodes.tolist()
        for m in range(len(nodes_ori)):
            index_nodes = nodes_tolist.index(nodes_ori[m])
            result[m] = X[index_nodes]
        return  result,all_weight


    def layer_node_attention(self, node_features, i):
        a_temp = torch.zeros(node_features.shape[1], 1) 
        a_temp[i, 0] = 1
        layer_attention = torch.transpose(node_features, 2, 1)
        b = layer_attention.matmul(a_temp).reshape(layer_attention.shape[0], layer_attention.shape[1])
        return b
    
    def layer_semantic(self,node_layer_feature):
        layer_semantic_ = torch.zeros(node_layer_feature.shape[0],self.metapath_number-1,2,self.features_num)
        for k in range(node_layer_feature.shape[0]):
            adj_pernode = node_layer_feature[k]
            temp_node = torch.zeros(self.metapath_number,2,self.features_num)
            temp_path = torch.zeros(2,self.features_num)
            temp_path[0] = adj_pernode[self.layer_predict]
            for j in range(self.metapath_number):
                if j==self.layer_predict:
                    continue
                temp_path[1] = adj_pernode[j]
                temp_node[j] = temp_path
            temp_node = temp_node[torch.arange(temp_node.size(0))!=self.layer_predict]
            layer_semantic_[k] =  temp_node
        return layer_semantic_

class BitwiseMultipyLogis(nn.Module):
    def __init__(self, features_num, dropout, alpha, metapath_number,device):
        super(BitwiseMultipyLogis, self).__init__()
        self.features_num = features_num
        self.metapath_number = metapath_number
        self.dropout = dropout
        self.alpha = alpha
        self.leakyReLU = nn.LeakyReLU(alpha)
        self.logis = LogisticVector(features_num,1)
        self.device = device
        self.trans = nn.Parameter(torch.eye(features_num))
        self.bias = nn.Parameter(torch.Tensor(features_num))
        nn.init.zeros_(self.bias)

    def forward(self, node_features, nodes_ori, layer_predict):
        nodes_ori = np.array(nodes_ori)
        nodes_ori = nodes_ori.tolist()
        nodes = torch.tensor(np.unique(nodes_ori))
        node_features_temp = torch.zeros((2,node_features[0].size(0),node_features[0].size(1))).to(self.device)
        for kl in range(self.metapath_number):
            node_features_temp[kl] = torch.tanh(torch.matmul(node_features[kl],self.trans)+self.bias)
        node_features = node_features_temp
        layer_all_attention = torch.transpose(node_features, 0, 1) 
        semantic_pernode = self.layer_bitwise(layer_all_attention,layer_predict)  
        Z = torch.zeros(node_features.shape[1], node_features.shape[2]).cuda(self.device)
        adj_node = layer_all_attention 
        weight = self.logis(semantic_pernode)
        weight = F.softmax(weight, dim=1)
        for kk in range(self.metapath_number):
            if kk == layer_predict:
                continue
            Z = Z + weight[:,kk].unsqueeze(1) * adj_node[:,kk]
        X = node_features[layer_predict] + Z
        return X

    def layer_node_attention(self, node_features, i):
        a_temp = torch.zeros(node_features.shape[1], 1).cuda(self.device)
        a_temp[i, 0] = 1
        layer_attention = torch.transpose(node_features, 2, 1)
        b = layer_attention.matmul(a_temp).reshape(layer_attention.shape[0], layer_attention.shape[1])
        return b

    def layer_bitwise(self, node_layer_feature,layer_predict):
        layer_semantic = torch.zeros_like(node_layer_feature).cuda(self.device)
        exclude_dims = [i for i in range(self.metapath_number) if i != layer_predict]
        excluded_features = node_layer_feature[:, exclude_dims] * node_layer_feature[:, layer_predict].unsqueeze(1)
        self_features =  node_layer_feature[:,layer_predict] * node_layer_feature[:, layer_predict]
        layer_semantic[:, exclude_dims] = excluded_features
        layer_semantic[:, layer_predict] = self_features
        return layer_semantic


class LogisticVector(torch.nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(LogisticVector, self).__init__()
        self.n_feature = n_feature
        self.parameter = torch.nn.Linear(n_feature, n_hidden)
        self.active = nn.Sigmoid()
    def forward(self,x):
        value = self.parameter(x)
        out = self.active(value)
        return out.squeeze()
    
class LogisticVector2(torch.nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(LogisticVector2, self).__init__()
        self.n_feature = n_feature
        self.parameter = torch.nn.Linear(n_feature, n_hidden) 
        self.active = nn.Sigmoid()
    def forward(self,x):
        value = self.parameter(x)
        out = self.active(value )
        return out.squeeze()
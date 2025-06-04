import torch
from torch.autograd import Variable
import torch.nn as nn
import pickle
import torch
import numpy as np
import networkx as nx
import torch
import copy
import argparse
import sys
import scipy.sparse as sp
from collections import defaultdict
import math

class LogisticRegression(torch.nn.Module):
    def __init__(self, n_feature, n_hidden):  
        super(LogisticRegression, self).__init__()  
        self.n_feature = n_feature
        self.parameter = torch.nn.Linear(n_feature, n_hidden) # hidden layer  
        self.active = nn.Sigmoid() ####  # output layer  
    def forward(self,x):
        node_num, out_features = x.size()
        even = [i for i in range(node_num) if divmod(i,2)[1] == 0]
        odd = [i for i in range(node_num) if divmod(i,2)[1] != 0]
        even_tensor = Variable(torch.LongTensor(even))
        odd_tensor = Variable(torch.LongTensor(odd))
        properity = torch.mul(torch.index_select(x,0,odd_tensor),torch.index_select(x,0,even_tensor))
        value = self.parameter(properity)
        out = self.active(value )
        return out.squeeze()





def iterate_return(idx,labels,adj, batch_num= 256):
    print("in iterate_return")
    #pre for batch, sample data,
    labels = labels.cpu().numpy().astype(int)
    node_num = np.shape(adj)[0]
    return_list = []
    for chunk_id, ids in enumerate(np.array_split(idx, batch_num)):
        targets = labels[ids.astype(int)]
        node_list = edge2pair(ids,node_num)
        train_num = len(node_list)
        node_list, targets = __batch_to_torch(np.array(node_list),targets)
        portion_half = int(math.ceil(train_num//2*0.9))
        portion = portion_half * 2
        son_return = []
        for i in  node_list[:portion], targets[:portion_half], node_list[portion:], targets[portion_half:]:
            son_return.append(i)
        return_list.append(son_return)
    
    print("iterate_return finished")
    return return_list

def __batch_to_torch(*args):
    return_list = []
    for cnt,v in enumerate(args):
        m = torch.Tensor(v)
        if len(m.size()) != 1:
            m = m.view(-1,1).squeeze()
        return_list.append(m)
    return return_list

def edge2pair(idx, node_num):
    node_list = []
    for i in idx:
        node_list.extend(divmod(i, node_num))
    return node_list

def accuracy(output, labels):
    assert output.size() == labels.size()
    output = output > 0.5
    preds = output.type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / float(len(labels))

def sample_test_batch(idx,labels,adj,batch_size = 1):
    node_num = np.shape(adj)[0]
    return_list = []
    for chunk_id, ids in enumerate(np.array_split(idx, batch_size)):
        targets = labels[ids.astype(int)]
        node_list = edge2pair(ids,node_num)
        train_num = len(node_list)
        node_list,targets = __batch_to_torch(np.array(node_list),targets)
        son_list = [node_list,targets]
        return_list.append(son_list)
    return return_list

import numpy as np
from GMM import GMM
import sys
from collections import defaultdict
import Mcc
import networkx as nx
import random
class Graph:
    def __init__(self, N = 0):
        # N number of initialized node
        self.num_nodes = N
        self.adj_list = []
        self.edge_list = []
        self.num_edges = []
        self.max_rank = 0
        num_nodes_layer1 = N
        num_nodes_layer2 = N
        if N != 0:
            link1,link2 = GMM(N)
            G1 = nx.Graph()
            G1.add_nodes_from(range(N))
            G1.add_edges_from(link1)
            G2 = nx.Graph()
            G2.add_nodes_from(range(N))
            G2.add_edges_from(link2)
            self.adj_list = [list(G1.adjacency()),list(G2.adjacency())]
            self.edge_list = [G1.edges(),G2.edges()]
            self.num_edges = [len(self.edge_list[0]),len(self.edge_list[1])]
            self.max_rank = 0
            self.weights =[{},{}]
            self.ori_rank(G1,G2)
            self.cal_degree(G1,G2)

    def ori_rank(self,G1,G2):
        remove_edge = [set(),set()]
        connected_components = Mcc.MCC(G1.copy(),G2.copy(),remove_edge)
        self.max_rank = Mcc.find_max_set_length(connected_components)
        
    def cal_degree(self, G1, G2):
        degree1 = nx.degree(G1)
        maxDegree1 = max(dict(degree1).values())
        degree2 = nx.degree(G2)
        maxDegree2 = max(dict(degree2).values())
        for node in G1.nodes():
            self.weights[0][node] = degree1[node]/maxDegree1
            # self.weights[0][node] = random.uniform(0,1)
            # self.weights[0][node] = 1
        for node in G2.nodes(): 
            self.weights[1][node] = degree2[node]/maxDegree2
            # self.weights[1][node] = random.uniform(0,1)
            # self.weights[1][node] = 1
class GSet:
    def __init__(self):
        self.graph_pool = {}

    def InsertGraph(self, gid, graph):
        assert gid not in self.graph_pool
        self.graph_pool[gid] = graph

    def Sample(self):
        assert self.graph_pool
        gid = np.random.choice(list(self.graph_pool.keys()))
        return self.graph_pool[gid]

    def Get(self, gid):
        assert gid in self.graph_pool
        return self.graph_pool[gid]

    def Clear(self):
        self.graph_pool.clear()
        
class Graph_test:
    def __init__(self,G1,G2):
        # N number of initialized node
        self.num_nodes = len(G1.nodes)
        self.adj_list = [list(G1.adjacency()),list(G2.adjacency())]
        self.edge_list = [G1.edges(),G2.edges()]
        self.num_edges = [len(self.edge_list[0]),len(self.edge_list[1])]
        self.max_rank = 0
        self.weights =[{},{}]
        self.ori_rank(G1,G2)
        if self.max_rank != 1:
            self.cal_degree(G1,G2)
        
    def ori_rank(self,G1,G2):
        remove_edge = [set(),set()]
        connected_components = Mcc.MCC(G1.copy(),G2.copy(),remove_edge)
        self.max_rank = Mcc.find_max_set_length(connected_components)
        return G1, G2
    
    def cal_degree(self, G1, G2):
        degree1 = nx.degree(G1)
        maxDegree1 = max(dict(degree1).values())
        degree2 = nx.degree(G2)
        maxDegree2 = max(dict(degree2).values())
        # normal_distribution_1 = np.random.normal(loc=0.5, scale=0.1, size=len(G1.nodes()))
        # normal_distribution_1 = np.clip(normal_distribution_1, 0, 1)
        
        # normal_distribution_2 = np.random.normal(loc=0.5, scale=0.1, size=len(G2.nodes()))
        # normal_distribution_2 = np.clip(normal_distribution_2, 0, 1)
        # poisson_distribution_1 = np.random.poisson(5, size=len(G1.nodes()))
        # poisson_distribution_2 = np.random.poisson(5, size=len(G2.nodes()))
        # poisson_distribution_1 = (poisson_distribution_1 - poisson_distribution_1.min()) / (poisson_distribution_1.max() - poisson_distribution_1.min())
        # poisson_distribution_2 = (poisson_distribution_2 - poisson_distribution_2.min()) / (poisson_distribution_2.max() - poisson_distribution_2.min())
        for i, node in enumerate(G1.nodes()):
            self.weights[0][node] = degree1[node]/maxDegree1
            #self.weights[0][node] = random.uniform(0,1)
            #self.weights[0][node] = normal_distribution_1[i]
            #self.weights[0][node] = poisson_distribution_1[i]
        for i, node in enumerate(G2.nodes()):
            self.weights[1][node] = degree2[node]/maxDegree2
            #self.weights[1][node] = random.uniform(0,1)
            #self.weights[1][node] = normal_distribution_2[i]
            #self.weights[1][node] = poisson_distribution_2[i]
            
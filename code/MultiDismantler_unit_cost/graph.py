import numpy as np
from GMM import GMM
import sys
from collections import defaultdict
import Mcc
import networkx as nx
class Graph:
    def __init__(self, N = 0):
        # N number of initialized nodes
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
            
    def reshape_graph(self, num_nodes, num_edges, edges_from, edges_to):
        self.num_nodes = num_nodes
        self.num_edges.append(num_edges)
        adj_list = [[] for _ in range(num_nodes)]
        edge_list = [(edges_from[i], edges_to[i]) for i in range(num_edges)]
        for i in range(num_edges):
            x, y = edges_from[i], edges_to[i]
            adj_list[x].append(y)
            adj_list[y].append(x)
        self.adj_list.append(adj_list)
        self.edge_list.append(edge_list)

    def ori_rank(self,G1,G2):
        remove_edge = [set(),set()]
        connected_components = Mcc.MCC(G1.copy(),G2.copy(),remove_edge)
        self.max_rank = Mcc.find_max_set_length(connected_components)
        
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
        
        self.num_nodes = len(G1.nodes)
        self.adj_list = [list(G1.adjacency()),list(G2.adjacency())]
        self.edge_list = [G1.edges(),G2.edges()]
        self.num_edges = [len(self.edge_list[0]),len(self.edge_list[1])]
        self.max_rank = 0
        self.weights =[{},{}]
        self.ori_rank(G1,G2)
        
    def ori_rank(self,G1,G2):
        remove_edge = [set(),set()]
        connected_components = Mcc.MCC(G1.copy(),G2.copy(),remove_edge)
        self.max_rank = Mcc.find_max_set_length(connected_components)
        return G1, G2
    
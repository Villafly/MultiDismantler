# mvc_env.py
from typing import List, Set
import random
from disjoint_set import DisjointSet
from graph import Graph
import networkx as nx
import Mcc
class MvcEnv:
    def __init__(self, norm):
       
        self.norm = norm  
        self.graph = Graph(0)
        self.numCoveredEdges = [0,0] 
        self.CcNum = 1.0  
        self.state_seq = []  
        self.act_seq = []  
        self.action_list = []  
        self.reward_seq = []  
        self.sum_rewards = []  
        self.covered_set = set()  
        self.avail_list = []  
        self.remove_edge = [set(),set()]
        self.state_seq_edges = []
        self.MaxCCList = [1]
        self.score = 0.0
        self.flag = 0
        self.G1 = None
        self.G2 = None
    def s0(self, _g: Graph):
        
        self.graph = _g  
        self.covered_set.clear()  
        self.action_list.clear() 
        self.numCoveredEdges = [0,0]
        self.CcNum = 1.0  
        self.state_seq.clear() 
        self.act_seq.clear()  
        self.reward_seq.clear() 
        self.sum_rewards.clear() 
        self.remove_edge[0].clear()
        self.remove_edge[1].clear()
        self.state_seq_edges.clear()
        self.MaxCCList = [1]
        self.score = 0.0
        self.flag = 0
        self.G1 = None
        self.G2 = None
        self.getMaxConnectedNodesNum()
    def step(self, a):
        assert self.graph
        assert a not in self.covered_set

        self.state_seq.append(self.action_list.copy())  
        remove_edge = [self.remove_edge[0].copy(),self.remove_edge[1].copy()]
        self.state_seq_edges.append(remove_edge)
        self.act_seq.append(a) 
        self.covered_set.add(a) 
        self.action_list.append(a)  
        for i in range(2):
            for neigh in self.graph.adj_list[i][a][1]:              
                    if neigh not in self.covered_set and (neigh,a) not in self.remove_edge[i]:
                        self.numCoveredEdges[i] += 1                    
        r_t, _ = self.getReward(a)  
        self.reward_seq.append(r_t) 
        self.sum_rewards.append(r_t)             
        return r_t

    def stepWithoutReward(self, a):
      
        assert self.graph
        assert a not in self.covered_set
        self.covered_set.add(a) 
        self.action_list.append(a)    
        for i in range(2):
            for neigh in list(self.graph.adj_list[i][a][1]):            
                    if neigh not in self.covered_set and (neigh,a) not in self.remove_edge[i]:
                        self.numCoveredEdges[i] += 1  
        r_t, remain_MCC_size = self.getReward(a)  
        self.score += -1 * r_t
        self.MaxCCList.append(remain_MCC_size)
        
    def randomAction(self):
        
        assert self.graph
        self.avail_list.clear()  
        for i in range(self.graph.num_nodes):
            if i not in self.covered_set:
                useful1 = any(neigh not in self.covered_set and (i,neigh) not in self.remove_edge[0] for neigh in self.graph.adj_list[0][i][1])
                useful2 = any(neigh not in self.covered_set and (i,neigh) not in self.remove_edge[1] for neigh in self.graph.adj_list[1][i][1])
                if useful1 and useful2:
                    self.avail_list.append(i)
        assert self.avail_list
        idx = random.choice(self.avail_list)  
        return idx

    def betweenAction(self):
        
        assert self.graph
        adj_dic_origin = {}
        adj_list_reID = []

        for i in range(self.graph.num_nodes):
            if i not in self.covered_set:
                for neigh in self.graph.adj_list[i]:
                    if neigh not in self.covered_set:
                        if i in adj_dic_origin:
                            adj_dic_origin[i].append(neigh)
                        else:
                            adj_dic_origin[i] = [neigh]

        id2node = {num: node for num, node in enumerate(adj_dic_origin)}
        node2id = {node: num for num, node in id2node.items()}
        adj_list_reID = [[node2id[neigh] for neigh in adj_dic_origin[node]] for node in adj_dic_origin]

        BC = self.betweenness(adj_list_reID) 
        maxID = max(range(len(BC)), key=BC.__getitem__)
        idx = id2node[maxID]  

        return idx

    def isTerminal(self):
        
        assert self.graph
        return self.graph.num_edges[0] == (self.numCoveredEdges[0] + len(self.remove_edge[0])/2) or self.graph.num_edges[1] == (self.numCoveredEdges[1] + len(self.remove_edge[1])/2)
        # return self.graph.max_rank <=sq_num
    
    def getReward(self, a):
       
        orig_node_num = float(self.graph.num_nodes)
        rank = self.getMaxConnectedNodesNum(a)
        total_weight0 = sum(self.graph.weights[0].values())
        total_weight1 = sum(self.graph.weights[1].values())
        return -rank / (self.graph.max_rank) * (self.graph.weights[0][a]/total_weight0 + self.graph.weights[1][a]/total_weight1)/2.0, rank / (self.graph.max_rank)
        # return -rank / (self.graph.max_rank * orig_node_num),  rank / (self.graph.max_rank)
    def getMaxConnectedNodesNum(self,a=None):
      
        assert self.graph
        if self.flag == 0 :
            self.G1 = nx.Graph()
            self.G2 = nx.Graph()
            self.G1.add_nodes_from(range(0,self.graph.num_nodes))
            self.G2.add_nodes_from(range(0,self.graph.num_nodes))
            for i,neighbors in self.graph.adj_list[0]:
                for j in neighbors:
                    if i not in self.covered_set and j not in self.covered_set and (i,j) not in self.remove_edge[0]:
                        self.G1.add_edge(i,j)
            for i,neighbors in self.graph.adj_list[1]:
                for j in neighbors:
                    if i not in self.covered_set and j not in self.covered_set and (i,j) not in self.remove_edge[1]:
                        self.G2.add_edge(i,j)
            self.flag = 1
        else :
            self.G1.remove_node(a)
            self.G2.remove_node(a)
        connected_components = Mcc.MCC(self.G1,self.G2,self.remove_edge)
        rank = Mcc.find_max_set_length(connected_components)
        return float(rank)

    def betweenness(self, adj_list):
       
        nvertices = len(adj_list)
        CB = [0.0] * nvertices
        norm = float((nvertices - 1) * (nvertices - 2))

        for i in range(nvertices):
            PredList = [[] for _ in range(nvertices)]
            d = [float('inf')] * nvertices
            d[i] = 0
            sigma = [0] * nvertices
            sigma[i] = 1
            delta = [0.0] * nvertices
            Q = [i]
            S = []

            while Q:
                u = Q.pop(0)
                S.append(u)
                for neigh in adj_list[u]:
                    v = neigh
                    if d[v] == float('inf'):
                        d[v] = d[u] + 1
                        Q.append(v)
                    if d[v] == d[u] + 1:
                        sigma[v] += sigma[u]
                        PredList[v].append(u)

            while S:
                u = S.pop()
                for pred in PredList[u]:
                    delta[pred] += (sigma[pred] / sigma[u]) * (1 + delta[u])
                if u != i:
                    CB[u] += delta[u]

            PredList.clear()
            d.clear()
            sigma.clear()
            delta.clear()

        for i in range(nvertices):
            if norm == 0:
                CB[i] = 0
            else:
                CB[i] = CB[i] / norm

        return CB

'''
def G2P(graph1: Graph) -> Graph:
    num_nodes = graph1.num_nodes
    num_edges = graph1.num_edges
    edge_list = graph1.edge_list
    cint_edges_from = [edge[0] for edge in edge_list]
    cint_edges_to = [edge[1] for edge in edge_list]
    return Graph(num_nodes, num_edges, cint_edges_from, cint_edges_to)
'''

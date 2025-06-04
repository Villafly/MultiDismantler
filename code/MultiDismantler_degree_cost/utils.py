# utils.py

from typing import List
from disjoint_set import DisjointSet
from graphutils import GraphUtil
import networkx as nx
import Mcc
class Utils:
    def __init__(self):
        self.MaxWccSzList = []

   
    def reInsert(self, graph, solution, allVex, decreaseStrategyID, reinsertEachStep):

        return self.reInsert_inner(solution, graph, allVex, decreaseStrategyID, reinsertEachStep)

    def reInsert_inner(self, beforeOutput, graph, allVex, decreaseStrategyID, reinsertEachStep):
        currentAdjListGraph = []
        backupCompletedAdjListGraph = graph.adj_list.copy()
        currentAllVex = [False] * graph.num_nodes

        for eachV in allVex:
            currentAllVex[eachV] = True

        leftOutput = set(beforeOutput)
        finalOutput = []
        disjoint_Set = DisjointSet(graph.num_nodes)
        graphutil = GraphUtil()

        while leftOutput:
            batchList = []
            for eachNode in leftOutput:
                decreaseValue = decreaseStrategyID.decreaseComponentNumIfAddNode(
                    backupCompletedAdjListGraph, currentAllVex, disjoint_Set, eachNode
                )
                batchList.append((decreaseValue, eachNode))

            batchList.sort()

            for i in range(min(reinsertEachStep, len(batchList))):
                finalOutput.append(batchList[i][1])
                leftOutput.remove(batchList[i][1])
                graphutil.recover_add_node(
                    backupCompletedAdjListGraph,
                    currentAllVex,
                    currentAdjListGraph,
                    batchList[i][1],
                    disjoint_Set,
                )

        finalOutput.reverse()
        return finalOutput

  
    def getRobustness(self, graph, solution):
        assert(graph)
        self.MaxWccSzList = [] 
        backupCompletedAdjListGraph = graph.adj_list.copy()   
        currentAdjList = [[],[]]  
        graphutil = GraphUtil()
        disjoint_Set1 = DisjointSet(graph.num_nodes) 
        disjoint_Set2 = DisjointSet(graph.num_nodes)  
        disjoint_Set = [disjoint_Set1,disjoint_Set2]
        backupAllVex = [False] * graph.num_nodes 
        totalMaxNum = 0.0 
        temp = 0.0 
        covered_nodes = []
        for Node in reversed(solution):
            graphutil.recover_add_node(
                backupCompletedAdjListGraph, backupAllVex, currentAdjList, Node, disjoint_Set
            )
            covered_nodes.append(Node)
            G1 = nx.Graph()
            G2 = nx.Graph()
            G1.add_nodes_from(range(0, graph.num_nodes))
            G2.add_nodes_from(range(0, graph.num_nodes))
            for i in covered_nodes:
                for j in graph.adj_list[0][i]:
                    #for k in range(len(currentAdjList[0])):
                    if len(currentAdjList[0]) >= i+1:
                        if j in currentAdjList[0][i]:
                            G1.add_edge(i, j)
            for i in covered_nodes:
                for j in graph.adj_list[1][i]:
                    #for k in range(len(currentAdjList[1])):
                    if len(currentAdjList[1]) >= i+1:
                        if j in currentAdjList[1][i]:
                            G2.add_edge(i, j)
            #print(G1.edges)
            remove_edge = [set(),set()]
            connected_components= Mcc.MCC(G1, G2, remove_edge)
            rank = Mcc.find_max_set_length(connected_components)
            #print(rank)
            totalMaxNum += rank
            self.MaxWccSzList.append(rank/graph.max_rank)
            temp = rank
        totalMaxNum = totalMaxNum - temp
        self.MaxWccSzList.reverse()
        return totalMaxNum / (graph.max_rank*graph.num_nodes)

   
    def getMxWccSz(self, graph):
        disjoint_Set = DisjointSet(graph.num_nodes)
        for i in range(len(graph.adj_list)):
            for j in range(len(graph.adj_list[i])):
                disjoint_Set.merge(i, graph.adj_list[i][j])
        return disjoint_Set.max_rank_count

 
    def Betweenness(self, graph):
        nvertices = graph.num_nodes
        CB = [0.0] * nvertices
        norm = (nvertices - 1) * (nvertices - 2)

        for i in range(nvertices):
            PredList = [[] for _ in range(nvertices)]
            d = [4294967295] * nvertices
            d[i] = 0
            sigma = [0] * nvertices
            sigma[i] = 1
            delta = [0.0] * nvertices
            Q = []
            S = []

            Q.append(i)

            while Q:
                u = Q.pop(0)
                S.append(u)

                for v in graph.adj_list[u]:
                    if d[v] == 4294967295:
                        d[v] = d[u] + 1
                        Q.append(v)

                    if d[v] == d[u] + 1:
                        sigma[v] += sigma[u]
                        PredList[v].append(u)

            while S:
                u = S.pop()
                for j in PredList[u]:
                    delta[j] += (sigma[j] / sigma[u]) * (1 + delta[u])

                if u != i:
                    CB[u] += delta[u]

            PredList = []
            d = []
            sigma = []
            delta = []

        for i in range(nvertices):
            CB[i] = CB[i] / norm
        print(CB)
        return CB

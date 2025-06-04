import numpy as np
from graph import Graph
from graph_struct import GraphStruct
import torch
import Mcc
import networkx as nx
from typing import List, Tuple, Dict
import sys
#   Preparation of batch data in graph neural networks
class SparseMatrix:
    def __init__(self):
        self.rowIndex = []
        self.colIndex = []
        self.value = []
        self.rowNum = 0
        self.colNum = 0

class PrepareBatchGraph:
    def __init__(self, aggregatorID):
        self.aggregatorID = aggregatorID
        self.act_select = [SparseMatrix(),SparseMatrix()]
        self.rep_global = [SparseMatrix(),SparseMatrix()]
        self.n2nsum_param = [SparseMatrix(),SparseMatrix()]
        self.laplacian_param = [SparseMatrix(),SparseMatrix()]
        self.subgsum_param = [SparseMatrix(),SparseMatrix()]
        self.idx_map_list = []
        self.subgraph_id_span = []
        self.aux_feat = []
        self.avail_act_cnt = []
        self.graph = [GraphStruct(),GraphStruct()]
        self.adj= []
        self.virtual_adj = []
        self.remove_edge_list = []
        
    def get_status_info(self,g: Graph,covered: List[int], remove_edge: List[set]):
        c = set(covered)
        idx_map = [[-1] * g.num_nodes, [-1] * g.num_nodes]
        counter = [0,0]
        twohop_number = [0,0]
        threehop_number = [0,0]
        node_twohop_set = [set(),set()]
        n = [0,0]
        node_twohop_counter = [{},{}] 
        isolated_nodes = []          
        for i in range(2):         
            for p in g.edge_list[i]: 
                if tuple(p) in remove_edge[i]:
                    continue                         
                ##if p[0] in c_new_set or p[1] in c_new_set:
                if p[0] in c or p[1] in c:
                    counter[i] += 1
                else:
                    if idx_map[i][p[0]] < 0:
                        n[i] += 1

                    if idx_map[i][p[1]] < 0:
                        n[i] += 1

                    idx_map[i][p[0]] = 0
                    idx_map[i][p[1]] = 0

                    if p[0] in node_twohop_counter[i]:
                        twohop_number[i] += node_twohop_counter[i][p[0]]
                        node_twohop_counter[i][p[0]] = node_twohop_counter[i][p[0]] + 1
                    else:
                        node_twohop_counter[i][p[0]] = 1

                    if p[1] in node_twohop_counter[i]:
                        twohop_number[i] += node_twohop_counter[i][p[1]]
                        node_twohop_counter[i][p[1]] = node_twohop_counter[i][p[1]] + 1
                    else:
                        node_twohop_counter[i][p[1]] = 1           
        assert idx_map[0] == idx_map[1]
        return n,counter,twohop_number,threehop_number,idx_map,remove_edge

    def Setup_graph_input(self, idxes, g_list, covered, actions, remove_edges):
        self.act_select = [SparseMatrix(),SparseMatrix()]
        self.rep_global = [SparseMatrix(),SparseMatrix()]
        self.idx_map_list = []
        self.avail_act_cnt = []

        node_cnt = [0,0]
        for i, idx in enumerate(idxes):
            g = g_list[idx]
            temp_feat1 = []
            temp_feat2 = []
            if remove_edges == None:
                avail, counter, twohop_number, _, idx_map, remove_edge = self.get_status_info(g, covered[i], remove_edges)
            else:
                avail, counter, twohop_number, _, idx_map, remove_edge = self.get_status_info(g, covered[i], remove_edges[i])
            
            if g.num_nodes > 0:
                temp_feat1.append(len(covered[i]) / g.num_nodes)
                temp_feat2.append(len(covered[i]) / g.num_nodes)
            temp_feat1.append(counter[0] / g.num_edges[0])
            temp_feat1.append(twohop_number[0] / (g.num_nodes * g.num_nodes))
            temp_feat1.append(1.0)
            temp_feat2.append(counter[1] / g.num_edges[1])
            temp_feat2.append(twohop_number[1] / (g.num_nodes * g.num_nodes))
            temp_feat2.append(1.0)
            temp_feat = [temp_feat1,temp_feat2]
            for j in range (2):
                node_cnt[j] += avail[j]

            self.aux_feat.append(temp_feat)
            self.idx_map_list.append(idx_map)
            self.avail_act_cnt.append(avail)
            self.remove_edge_list.append(remove_edge)
            
        for j in range(2):
            self.graph[j].resize(len(idxes), node_cnt[j])

            if actions:
                self.act_select[j].rowNum = len(idxes)
                self.act_select[j].colNum = node_cnt[j]
            else:
                self.rep_global[j].rowNum = node_cnt[j]
                self.rep_global[j].colNum = len(idxes)

        node_cnt = [0,0]
        edge_cnt = [0,0]

        for i, idx in enumerate(idxes):
            g = g_list[idx]
            idx_map = self.idx_map_list[i]
            remove_edge = self.remove_edge_list[i]
            t = [0,0]
            for j in range(g.num_nodes):
                for h in range(2):
                    if idx_map[h][j] < 0:
                        continue
                    idx_map[h][j] = t[h]
                    self.graph[h].add_node(i, node_cnt[h] + t[h])
                    if not actions:
                        self.rep_global[h].rowIndex.append(node_cnt[h] + t[h])
                        self.rep_global[h].colIndex.append(i)
                        self.rep_global[h].value.append(1.0)
                    t[h] += 1
            #error
            assert t[0] == self.avail_act_cnt[i][0]

            if actions:
                act = actions[idx]
                #error
                assert idx_map[0][act] >= 0 and act >= 0 and act < g.num_nodes
                for j in range(2):
                    self.act_select[j].rowIndex.append(i)
                    self.act_select[j].colIndex.append(node_cnt[j] + idx_map[j][act])
                    self.act_select[j].value.append(1.0)

            for j in range(2):
                for p in g.edge_list[j]:
                    if tuple(p) in remove_edge[j]:
                        continue 
                    if idx_map[j][p[0]] >= 0 and idx_map[j][p[1]] >= 0:
                        x, y = idx_map[j][p[0]] + node_cnt[j], idx_map[j][p[1]] + node_cnt[j]
                        self.graph[j].add_edge(edge_cnt[j], x, y)
                        edge_cnt[j] += 1
                        self.graph[j].add_edge(edge_cnt[j], y, x)
                        edge_cnt[j] += 1

                node_cnt[j] += self.avail_act_cnt[i][j]
        #error
        assert node_cnt[0] == self.graph[0].num_nodes
        result_list = self.n2n_construct(self.aggregatorID)
        self.n2nsum_param = result_list[0]
        self.laplacian_param = result_list[1]
        self.adj = result_list[2]
        result_list1 = self.subg_construct()
        self.subgsum_param = result_list1[0]
        self.virtual_adj = result_list1[1]
        for j in range(2):
            self.act_select[j] = self.convert_sparse_to_tensor(self.act_select[j])
            self.rep_global[j] = self.convert_sparse_to_tensor(self.rep_global[j])
            self.n2nsum_param[j] = self.convert_sparse_to_tensor(self.n2nsum_param[j])
            self.laplacian_param[j] = self.convert_sparse_to_tensor(self.laplacian_param[j])
            self.subgsum_param[j] = self.convert_sparse_to_tensor(self.subgsum_param[j])


    def SetupTrain(self, idxes, g_list, covered, actions, remove_edges):
        self.Setup_graph_input(idxes, g_list, covered, actions, remove_edges)

    def SetupPredAll(self, idxes, g_list, covered, remove_edges):
        self.Setup_graph_input(idxes, g_list, covered, None, remove_edges)
    '''
    def convert_sparse_to_tensor(self, matrix):
        indices = np.column_stack((matrix.rowIndex, matrix.colIndex))
        return torch.sparse.FloatTensor(torch.LongTensor(indices).t(), torch.FloatTensor(matrix.value),
                                         torch.Size([matrix.rowNum, matrix.colNum]))
    '''

    def convert_sparse_to_tensor(self, matrix):
        rowIndex= matrix.rowIndex
        colIndex= matrix.colIndex
        data= matrix.value
        rowNum= matrix.rowNum
        colNum= matrix.colNum
        indices = np.mat([rowIndex, colIndex]).transpose()

        index = torch.tensor(np.transpose(np.array(indices)))
        value = torch.Tensor(np.array(data))
        #index, value = torch_sparse.coalesce(index, value, m=rowNum, n=colNum)
        return_dict = {"index": index, "value": value, "m":rowNum, "n":colNum}
        return return_dict

    '''
    def graph_resize(self, size, node_cnt):
        self.graph = Graph(size, node_cnt)

    def graph_add_node(self, i, node):
        self.graph.add_node(i, node)

    def graph_add_edge(self, edge, x, y):
        self.graph.add_edge(edge, x, y)
    '''

    def n2n_construct(self, aggregatorID):
        result = [SparseMatrix(),SparseMatrix()]
        result_laplacian = [SparseMatrix(),SparseMatrix()]
        adj_matrixs = []
        for h in range(2):
            result[h].rowNum = self.graph[h].num_nodes
            result[h].colNum = self.graph[h].num_nodes
            result_laplacian[h].rowNum = self.graph[h].num_nodes
            result_laplacian[h].colNum = self.graph[h].num_nodes

            for i in range(self.graph[h].num_nodes):
                list1 = self.graph[h].in_edges.head[i]

                if len(list1) > 0:
                    result_laplacian[h].value.append(len(list1))
                    result_laplacian[h].rowIndex.append(i)
                    result_laplacian[h].colIndex.append(i)

                for j in range(len(list1)):
                    if aggregatorID == 0:
                        result[h].value.append(1.0)
                    elif aggregatorID == 1:
                        result[h].value.append(1.0 / len(list1))
                    elif aggregatorID == 2:
                        #neighborDegree = len(self.graph.in_edges.head[list1[j].second])
                        neighborDegree = len(self.graph[h].in_edges.head[list1[j][1]])
                        selfDegree = len(list1)
                        norm = np.sqrt(neighborDegree + 1) * np.sqrt(selfDegree + 1)
                        result[h].value.append(1.0 / norm)

                    result[h].rowIndex.append(i)
                    #result[i].colIndex.append(list1[j].second)
                    result[h].colIndex.append(list1[j][1])
                    result_laplacian[h].value.append(-1.0)
                    result_laplacian[h].rowIndex.append(i)
                    #result[i].result_laplacian[i].colIndex.append(list1[j].second)
                    result_laplacian[h].colIndex.append(list1[j][1])

            adj_matrix = np.zeros((self.graph[h].num_nodes,self.graph[h].num_nodes))
            for edge in self.graph[h].edge_list:
                i,j=edge
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1
            adj_matrixs.append(adj_matrix)
        return [result,result_laplacian,adj_matrixs]

    '''
    def e2n_construct(self):
        result = SparseMatrix()
        result.rowNum = self.graph.num_nodes
        result.colNum = self.graph.num_edges

        for i in range(self.graph.num_nodes):
            list1 = self.graph.in_edges.head[i]
            for j in range(len(list1)):
                result.value.append(1.0)
                result.rowIndex.append(i)
                result.colIndex.append(list1[j].first)
        return result

    def n2e_construct(self):
        result = SparseMatrix()
        result.rowNum = self.graph.num_edges
        result.colNum = self.graph.num_nodes

        for i in range(self.graph.num_edges):
            result.value.append(1.0)
            result.rowIndex.append(i)
            result.colIndex.append(self.graph.edge_list[i].first)

        return result

    def e2e_construct(self):
        result = SparseMatrix()
        result.rowNum = self.graph.num_edges
        result.colNum = self.graph.num_edges

        for i in range(self.graph.num_edges):
            node_from, node_to = self.graph.edge_list[i]
            list1 = self.graph.in_edges.head[node_from]

            for j in range(len(list1)):
                if list1[j].second == node_to:
                    continue
                result.value.append(1.0)
                result.rowIndex.append(i)
                result.colIndex.append(list1[j].first)

        return result
    '''

    def subg_construct(self):
        result = [SparseMatrix(),SparseMatrix()]
        virtual_adjs = []
        for h in range(2):
            result[h].rowNum = self.graph[h].num_subgraph
            result[h].colNum = self.graph[h].num_nodes

            subgraph_id_span = []
            start = 0
            end = 0

            for i in range(self.graph[h].num_subgraph):
                list1 = self.graph[h].subgraph.head[i]
                end = start + len(list1) - 1

                for j in range(len(list1)):
                    result[h].value.append(1.0)
                    result[h].rowIndex.append(i)
                    result[h].colIndex.append(list1[j])

                if len(list1) > 0:
                    subgraph_id_span.append((start, end))
                else:
                    subgraph_id_span.append((self.graph[h].num_nodes, self.graph[h].num_nodes))
                start = end + 1
            virtual_adj = np.zeros((result[h].rowNum,result[h].colNum))
            for i in range(len(result[h].value)):
                row_idx = result[h].rowIndex[i]
                col_idx = result[h].colIndex[i]
                weight = result[h].value[i]
                virtual_adj[row_idx][col_idx] = weight
            virtual_adjs.append(virtual_adj)
        return [result,virtual_adjs]



#coding=utf-8
import networkx as nx
import math
import random
from random import shuffle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import os,sys
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

os.chdir(sys.path[0])

def deledge(graph, index_map):
    for (u, v) in graph.edges:
        if u in index_map and v in index_map and index_map[u] == index_map[v]:
            continue
        else:
            graph.remove_edge(u,v)

def find_max_set_length(set_list):
    return len(max(set_list, key=len, default=0))

def find_set_length(set_list):
    set_lengths = []
    for integer_set in set_list:
        set_length = len(integer_set)
        set_lengths.append(set_length)
    return set_lengths

def find_connected_components(graph):
    connected_components = [set(component) for component in nx.connected_components(graph)]
    index_map = {}
    for i, component in enumerate(connected_components):
        for node in component:
            index_map[node] = i
    return connected_components, index_map

def MCC(G1, G2):
    connected_components1,index_map1 = find_connected_components(G1)
    connected_components2,index_map2 = find_connected_components(G2)
    while connected_components1 != connected_components2:
        deledge(G2, index_map1)
        connected_components2, index_map2 = find_connected_components(G2)
        deledge(G1, index_map2)
        connected_components1, index_map1 = find_connected_components(G1)
    return find_max_set_length(connected_components1)

def basic_ci(graph, node, degrees):
    ci = 0
    neighbors = list(graph.neighbors(node))
    node_degree = degrees[node]
    if node_degree == 0:
        ci = -1
    else :
        for neighbor in neighbors:
            ci += degrees[neighbor] - 1

        ci *= (node_degree - 1)
    return ci

def get_ci_dict(graph):
    degrees = dict(graph.degree())  
    ci_values = {node: basic_ci(graph, node, degrees) for node in graph.nodes()}
    return ci_values

def delnode(G1,G2,N):
    ci1 = get_ci_dict(G1)
    ci2 = get_ci_dict(G2)
    max_cs = []
    ND_temp_list = []
    target_index_val = []
    for node in ci1.keys():
        c1 = ci1[node]
        c2 = ci2[node]
        max_cs.append((node, max(c1,c2)))
        
    sorted_c = sorted(max_cs, key=lambda t: t[1], reverse=True)
    target_index_val = [sorted_c[0][0]]
    
    for node in target_index_val:
        G1.remove_node(node)
        G2.remove_node(node)
        ND_temp= MCC(G1, G2)
        ND_temp_list.append(ND_temp)
    return target_index_val,ND_temp_list

def critical_number(G1, G2, N, ND_ori):
    ND_temp = ND_ori
    i = 0
    solutions = []
    ND_mcc = [1]
    num = N
    score = 0.0
    new_num = math.sqrt(N) / ND_ori
    # while ND_temp != 1:
    while ND_mcc[-1] > new_num:
        deleteNode,ND_temp_list = delnode(G1, G2, N)
        for delete_node in deleteNode:
            solutions.append(delete_node)
        #ND_temp= MCC(G1, G2)
        for ND_temp in ND_temp_list:
            ND_mcc.append(ND_temp/ND_ori)
            score +=  ND_temp / (ND_ori * N)
    value_cost = len(solutions) / N
    return ND_mcc, solutions,score , value_cost

def read_multiplex(path, N):
    layers_matrix = []
    graphs = []
    _ii = []
    _jj = []
    _ww = []

    g = nx.Graph()
    for i in range(0, N):
        g.add_node(i)
    with open(path, "r") as lines:
        cur_id = 1
        for l in lines:
            elems = l.strip(" \n").split(" ")
            layer_id = int(elems[0])
            if cur_id != layer_id:
                adj_matr = nx.adjacency_matrix(g)
                layers_matrix.append(adj_matr)
                graphs.append(g)
                g = nx.Graph()

                for i in range(0, N):
                    g.add_node(i)

                cur_id = layer_id
            node_id_1 = int(elems[1]) - 1
            node_id_2 = int(elems[2]) - 1
            if node_id_1 == node_id_2:
                continue
            g.add_edge(node_id_1, node_id_2)

    adj_matr = nx.adjacency_matrix(g)
    layers_matrix.append(adj_matr)
    graphs.append(g)
    return layers_matrix, graphs

if __name__ == "__main__":

    #test_name = ['30-50','50-100','100-200','200-300','300-400','400-500']
    test_name = ['32','64','128','256','512','1024']
    dirt_type = ['data_g', 'data_gamma', 'data_k']
    #test_name =['1024']
    for dirt in dirt_type:
        file_path = f'../../results/CI/synthetic_cost/{dirt}'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for data in test_name:
            score = []
            value_cost_list = []
            for i in tqdm(range(20)):
                adj1 = np.load(f"../../../FINDER_CN_weight/synthetic/{dirt}/syn_%s/adj1_%s.npy"%(data,i))
                adj2 = np.load(f"../../../FINDER_CN_weight/synthetic/{dirt}/syn_%s/adj2_%s.npy"%(data,i))
                N = len(adj1)
                G1 = nx.Graph()
                G2 = nx.Graph()
                G1.add_nodes_from(range(0, N))
                G2.add_nodes_from(range(0, N))
                for i in range(0, N):
                    for j in range(0, N):
                        if adj1[i][j] == 1:
                            G1.add_edge(i, j)
                for i in range(0, N):
                    for j in range(0, N):
                        if adj2[i][j] == 1:
                            G2.add_edge(i, j)
                M_ori = MCC(G1, G2) 
                MCCs,remove_nodes,temp_score, value_cost = critical_number(G1.copy(), G2.copy(), N, M_ori)
                #temp_score = (sum(MCCs)-1)/N + (N-len(remove_nodes)-1) / (M_ori * N)
                score.append(temp_score)
                value_cost_list.append(value_cost)
            cost_mean = np.mean(value_cost_list)
            score_mean = np.mean(score)
            score_std = np.std(score)
            print(score_mean*100,score_std*100)
            with open('%s/test_max_result_%s_unit_cost.txt'%(file_path,data), 'w') as fout:
                # fout.write('%.4f±%.2f,' % (score_mean , score_std ))
                fout.write('%.4f' % (cost_mean))
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

def delnode(G1,G2,N):
    D1 = dict(G1.degree())
    D2 = dict(G2.degree())
    ND_temp_list = []
    max_degrees = []
    target_index_val = []
    flag = 0
    for node in D1.keys():
        degree1 = D1[node]
        degree2 = D2[node]
        max_degrees.append((node, max(degree1, degree2)))

    sorted_degree = sorted(max_degrees, key=lambda t: t[1], reverse=True)
    target_index_val = [sorted_degree[0][0]]
    for node in target_index_val:
        G1.remove_node(node)
        G2.remove_node(node)
        ND_temp= MCC(G1, G2)
        ND_temp_list.append(ND_temp)
    return target_index_val,flag,ND_temp_list

def critical_number(G1, G2, N, ND_ori):
    ND_temp = ND_ori
    i = 0
    solutions = []
    ND_mcc = [1]
    num = N
    score = 0.0
    new_num = math.sqrt(N)/ND_ori
    # while ND_temp != 1:
    #while num != 1:
    while ND_mcc[-1] > new_num:
        deleteNode,flag,ND_temp_list = delnode(G1, G2, N)
        if flag:
            break
        for delete_node in deleteNode:
            solutions.append(delete_node)
        #ND_temp= MCC(G1, G2)
        for ND_temp in ND_temp_list:
            ND_mcc.append(ND_temp/ND_ori)
            score +=  ND_temp / (ND_ori * N)
    value_cost = len(solutions) / N
    return ND_mcc, solutions, score , value_cost


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
    file_path = '../../../results/HDA/synthetic_k'
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    test_name = ['32','64','128','256','512','1024']
    dirt_type = ['data_g', 'data_gamma', 'data_k']
    #test_name =['1024']
    for dirt in dirt_type:
        file_path = f'../../results/HDA/synthetic_cost/{dirt}'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for data in test_name :
            score = []
            value_cost_list = []
            for i in tqdm(range(20)):
                # adj1 = np.load("../../data_k/syn_%s/adj1_%s.npy"%(data,i))
                # adj2 = np.load("../../data_k/syn_%s/adj2_%s.npy"%(data,i))
                adj1 = np.load(f"../../../FINDER_CN_weight/synthetic/{dirt}/syn_%s/adj1_%s.npy" % (data, i))
                adj2 = np.load(f"../../../FINDER_CN_weight/synthetic/{dirt}/syn_%s/adj2_%s.npy" % (data, i))
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
            with open('%s/test_2max_result_%s_unit_cost.txt'%(file_path,data), 'w') as fout:
                # fout.write('%.4f±%.2f,' % (score_mean , score_std ))
                fout.write('%.4f '% (cost_mean) )


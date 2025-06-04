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

def basic_ci(graph, node):
    ci = 0
    neighbors = list(graph.neighbors(node))
    for neighbor in neighbors:
        ci += graph.degree(neighbor) - 1

    ci *= (graph.degree(node) - 1)
    return ci

def get_ci_dict(graph):
    ci_values = {}
    for node in graph.nodes():
        ci_values[node] = basic_ci(graph, node)
    return ci_values

def delnode(G1,G2):
    ci1 = get_ci_dict(G1)
    ci2 = get_ci_dict(G2)

    max_cs = []
    for node in ci1.keys():
        c1 = ci1[node]
        c2 = ci2[node]
        #max_cs.append((node, c1+c2))
        max_cs.append((node, c1 + c2))
    # print(max_degrees)
    # ND=max(max_degrees, key=lambda t:t[1])
    max_c = max(max_cs, key=lambda t: t[1])[1]
    max_c_nodes = [t[0] for t in max_cs if t[1] == max_c]
    # print(len(max_degree_nodes))
    random_node = random.choice(max_c_nodes)
    G1.remove_node(random_node)
    G2.remove_node(random_node)

    return random_node

def critical_number(G1, G2, N, ND_ori):
    ND_temp = ND_ori
    i = 0
    solutions = []
    ND_mcc = [1]
    num = N
    #while ND_temp != 1:
    while num != 1:
        deleteNode = delnode(G1, G2)
        solutions.append(deleteNode)
        ND_temp = MCC(G1, G2)
        ND_mcc.append(ND_temp/ND_ori)
        num -= 1
    return ND_mcc, solutions

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

node_num = {'Padgett-Florentine-Families_multiplex': 16,
            'AirTrain': 69,  # [(1,2)]
            'Brain': 90,  # [(1,2)]
            # 'fao_trade_multiplex': 214,
            'Phys': 246,  # [(1,2), (1,3), (2,3)]
            'celegans_connectome_multiplex': 279,  # [(1,2), (1,3), (2,3)]
            # 'HumanMicrobiome_multiplex': 305,
            # 'xenopus_genetic_multiplex': 416,
            # 'pierreauger_multiplex': 514,
            'rattus_genetic_multiplex': 2640,  # [(1,2)]
            'sacchpomb_genetic_multiplex': 4092,  # [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4),(3,5)]
            'drosophila_genetic_multiplex': 8215,  # [(1,2)]
            'arxiv_netscience_multiplex': 14489,  # [(1,2),(1,4),(1,5),(2,4),(2,5),(2,6),(2,8),(3,4)]
            'Internet': 4202010733}

nums_dict = {'AirTrain': [(1,2)],
             'Brain': [(1,2)],
             'Phys': [(2,3)],  # [(1,2), (1,3), (2,3)],
             'celegans_connectome_multiplex': [(2,3)],
             'rattus_genetic_multiplex': [(1,2)],
             'sacchpomb_genetic_multiplex': [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4),(3,5)],
             'drosophila_genetic_multiplex': [(1,2)],
             'arxiv_netscience_multiplex': [(1,2),(1,4),(1,5),(2,4),(2,5),(2,6),(2,8),(3,4)]}

if __name__ == "__main__":
    datapath = '../../../data/'
    dataname = 'HumanMicrobiome_multiplex'
    #N = node_num[dataname]
    N = 305
    datafolder = datapath + dataname
    layers_matrix, graphs = read_multiplex(datafolder + '.edges', N)
    G1 = graphs[4]
    G2 = graphs[13]
    M_ori = MCC(G1, G2) 
    result_list_score = []
    Mcc_average = [0] * N

    n = 10
    for i in tqdm(range(n)):
        MCCs,remove_nodes = critical_number(G1.copy(), G2.copy(), N, M_ori)
        Mcc_average=[Mcc_average[i]+MCCs[i] for i in range(min(len(Mcc_average),len(MCCs)))]
        temp_score = (sum(MCCs)-1)/N + (N-len(remove_nodes)-1) / (M_ori * N)
        result_list_score.append(temp_score)
    score_mean = np.mean(result_list_score)
    score_std = np.std(result_list_score)
    print('HDA score:',score_mean,'std',score_std)  
    file_path1 = "../../../results/CI/none_cost/"+ "remove_nodes" + dataname + "_514_2.txt"
    with open(file_path1, 'w') as file:
        for c in remove_nodes:
            file.write(str(c) + '\n')
    file_path2 = "../../../results/CI/none_cost/MaxCCList_Strategy_"+ dataname + "_514_2.txt"
    with open(file_path2, 'w') as f_out:
            for j in range(N):
                if j < len(remove_nodes):           
                    f_out.write('%.8f\n' % (float(Mcc_average[j])/10.0))
                else:
                    Mcc = 1/M_ori
                    f_out.write('%.8f\n' % Mcc)
                    
    with open(file_path2, 'a') as f_out:
            f_out.write('%.8f\n' % score_mean)
            f_out.write('%.8f\n' % score_std)

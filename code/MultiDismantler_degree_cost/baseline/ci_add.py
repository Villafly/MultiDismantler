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
sys.path.append('..')
import graph

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
    for neighbor in neighbors:
        ci += degrees[neighbor] - 1

    ci *= (node_degree - 1)
    return ci

def get_ci_dict(graph):
    degrees = dict(graph.degree())  # 缓存所有节点的度数
    ci_values = {node: basic_ci(graph, node, degrees) for node in graph.nodes()}
    return ci_values

def delnode(G1,G2):
    ci1 = get_ci_dict(G1)
    ci2 = get_ci_dict(G2)

    max_cs = []
    for node in ci1.keys():
        c1 = ci1[node]
        c2 = ci2[node]
        #max_cs.append((node, c1+c2))
        max_cs.append((node, c1+c2))
    # print(max_degrees)
    # ND=max(max_degrees, key=lambda t:t[1])
    max_c = max(max_cs, key=lambda t: t[1])[1]
    max_c_nodes = [t[0] for t in max_cs if t[1] == max_c]
    # print(len(max_degree_nodes))
    random_node = random.choice(max_c_nodes)
    G1.remove_node(random_node)
    G2.remove_node(random_node)

    return random_node

def critical_number(G1, G2, N, ND_ori,w1,w2):
    ND_temp = ND_ori
    i = 0
    solutions = []
    ND_mcc = [1]
    num = N
    score = 0.0
    cost_list = []
    total_weight1 = sum(w1.values())
    total_weight2 = sum(w2.values())
    while ND_temp != 1:
    #while num != 1:
        print(num)
        deleteNode = delnode(G1, G2)
        solutions.append(deleteNode)
        ND_temp = MCC(G1, G2)
        ND_mcc.append(ND_temp/ND_ori)
        cost = (w1[solutions[-1]]/total_weight1 + w2[solutions[-1]]/total_weight2)/2.0
        score +=  ND_temp / ND_ori * cost
        cost_list.append(cost)
        num -= 1
    return ND_mcc, solutions, score, cost_list


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
    random.seed(0)
    datapath = '../../data/'
    dataname = 'Sanremo2016_final_multiplex'
    N = 56562
    datafolder = datapath + dataname
    layers_matrix, graphs = read_multiplex(datafolder + '.edges', N)
    G1 = graphs[1]
    G2 = graphs[2]
                
    degree1 = nx.degree(G1)
    maxDegree1 = max(dict(degree1).values())
    weights1 = {}
    for node in G1.nodes():
        #weights1[node] = degree1[node]/maxDegree1
        weights1[node] = random.uniform(0,1)
    
    degree2 = nx.degree(G2)
    maxDegree2 = max(dict(degree2).values())
    weights2 = {}
    for node in G2.nodes():
        #weights2[node] = degree2[node]/maxDegree2
        weights2[node] = random.uniform(0,1)
        
    total_weight1 = sum(weights1.values())
    total_weight2 = sum(weights2.values())
        
    
    M_ori = MCC(G1, G2) 
    result_list_score = []
    #Mccs_average = [0] * N
    #cost_average = [0] * N
    n = 1
    best_socre = float('inf')
    for i in tqdm(range(n)):
        MCCs,remove_nodes,score,cost_list = critical_number(G1.copy(), G2.copy(), N, M_ori,weights1,weights2)
        #Mccs_average=[Mccs_average[i]+MCCs[i] for i in range(min(len(Mccs_average),len(MCCs)))]
        #cost_average=[cost_average[i]+cost_list[i] for i in range(min(len(cost_average),len(cost_list)))]
    
        cost = []
        total_cost = 0
        remain_score = 0.0
        cost.append(total_cost)
        nodes = list(range(N))
        remain_nodes = list(set(nodes)^set(remove_nodes))
        for node in remove_nodes + remain_nodes[:-1]:
            total_cost += (weights1[node]/total_weight1+weights2[node]/total_weight2)/2.0
            cost.append(total_cost)
        # for node in remain_nodes[:-1]:
        #     remain_score += 1/ M_ori * (weights1[node]/total_weight1 + weights2[node]/total_weight2)/2.0
        # total_score = remain_score + score
        print(score)
        if score < best_socre:
            best_socre = score
        #result_list_score.append(total_score)
    #     result_list_score.append(score)
    # score_mean = np.mean(result_list_score)
    # score_std = np.std(result_list_score)
    # print('HDA score:',score_mean,'std',score_std)  
    # cost.append(score_mean) 
    # cost.append(score_std) 
            cost.append(score)    
            file_path1 = "../../results/CI/cost/"+ "cost_" + dataname + "_23_2.txt"
            with open(file_path1, 'w') as file:
                for c in cost:
                    file.write('%.8f\n' % c)
            file_path2 = "../../results/CI/cost/MaxCCList_Strategy_"+ dataname + "_23_2.txt"
            with open(file_path2, 'w') as f_out:
                    for j in range(N):
                        if j < len(remove_nodes):
                            f_out.write('%.8f\n' % MCCs[j])
                        else:
                            Mcc = 1/M_ori
                            f_out.write('%.8f\n' % Mcc)
            
            file_path3 = "../../results/CI/cost/"+ "remove_nodes" + dataname + "_23_2.txt"
            with open(file_path3, 'w') as file:
                for c in remove_nodes:
                    file.write(str(c) + '\n')
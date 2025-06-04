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
        max_cs.append((node, max(c1,c2)))
    # print(max_degrees)
    # ND=max(max_degrees, key=lambda t:t[1])
    max_c = max(max_cs, key=lambda t: t[1])[1]
    max_c_nodes = [t[0] for t in max_cs if t[1] == max_c]
    # print(len(max_degree_nodes))
    random_node = max_c_nodes[0]
    G1.remove_node(random_node)
    G2.remove_node(random_node)

    return random_node

def critical_number(G1, G2, N, ND_ori,w1,w2):
    ND_temp = ND_ori
    i = 0
    solutions = []
    ND_mcc = [1]
    nodes = G1.nodes()
    num =N
    # num = math.sqrt(N) / ND_ori
    score = 0.0
    cost_list = []
    total_weight1 = sum(w1.values())
    total_weight2 = sum(w2.values())

    # while ND_mcc[-1] > num:
    while ND_temp != 1:
    # while num != 1:
        deleteNode = delnode(G1, G2)
        solutions.append(deleteNode)
        ND_temp = MCC(G1, G2)
        ND_mcc.append(ND_temp/ND_ori)
        cost = (w1[solutions[-1]]/total_weight1 + w2[solutions[-1]]/total_weight2)/2.0
        score +=  ND_temp / ND_ori * cost
        cost_list.append(cost)
        num -= 1
    remain_node = list(set(nodes) ^ set(solutions))
    for cost_node in remain_node[:-1]:
        cost_list.append((w1[cost_node] / total_weight1 + w2[cost_node] / total_weight2) / 2.0)
    total_cost = np.sum(cost_list)
    return ND_mcc, solutions, score, cost_list, total_cost


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

    dir_type = ['data_gamma', 'data_g', 'data_k']

    # test_name = ['30-50','50-100','100-200','200-300','300-400','400-500']
    test_name = ['32', '64', '128', '256', '512', '1024']
    for dirt in dir_type:
        file_path = f'./results/CI/synthetic_cost/{dirt}_1'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for data in test_name:
            random.seed(0)
            score_list = []
            total_cost_list =[]
            for i in tqdm(range(20)):
                # adj1 = np.load("../../FINDER_CN/data/syn_%s/adj1_%s.npy"%(data,i))
                # adj2 = np.load("../../FINDER_CN/data/syn_%s/adj2_%s.npy"%(data,i))
                adj1 = np.load(f"../synthetic/{dirt}/syn_%s/adj1_%s.npy"%(data,i))
                adj2 = np.load(f"../synthetic/{dirt}/syn_%s/adj2_%s.npy"%(data,i))
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

                weights1 = {}
                degree1 = nx.degree(G1)
                maxDegree1 = max(dict(degree1).values())
                degree2 = nx.degree(G2)
                maxDegree2 = max(dict(degree2).values())
                if maxDegree1 == 0 or maxDegree2 == 0:
                    continue
                for node in G1.nodes():
                    weights1[node] = degree1[node] / maxDegree1
                    # weights1[node] = random.uniform(0,1)

                weights2 = {}
                for node in G2.nodes():
                    weights2[node] = degree2[node] / maxDegree2
                    # weights2[node] = random.uniform(0,1)

                if maxDegree1 == 0 or maxDegree2 == 0:
                    continue

                M_ori = MCC(G1, G2) 
                MCCs,remove_nodes,score,cost_list, total_cost = critical_number(G1.copy(), G2.copy(), N, M_ori,weights1,weights2)
                # remain_score = 0.0
                # nodes = list(range(N))
                # remain_nodes = list(set(nodes)^set(remove_nodes))
                # for node in remain_nodes[:-1]:
                #     remain_score += 1/ M_ori * (weights1[node]/total_weight1 + weights2[node]/total_weight2)/2.0
                # total_score = remain_score + score
                score_list.append(score)
                total_cost_list.append(total_cost)
            cost_mean = np.mean(total_cost_list)
            score_mean = np.mean(score_list)
            score_std = np.std(score_list)
            # print(score_mean*100,score_std*100)
            print(score_mean, score_std)
            with open('%s/2max_result_%s_new.txt'%(file_path,data), 'w') as fout:
                fout.write('%.4f±%.2f,' % (score_mean, score_std ))
                # fout.write('%.4f' % (cost_mean))

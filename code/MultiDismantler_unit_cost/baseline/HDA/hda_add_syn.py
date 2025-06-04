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

def add_edges_from_file(graph, file):
    for line in file:
        edge = line.strip().split(' '.encode())
        u, v = int(edge[0]), int(edge[1])
        graph.add_edge(u, v)
    return graph

def find_connected_components(graph):
    connected_components = []
    #idx_map = [-1] * len(graph.nodes)
    for node in graph.nodes:
        # if idx_map[node] >=0 :
        #     continue
        connected_component = nx.node_connected_component(graph, node)
        # for n in connected_component:
        #     idx_map[n] = 0
        if connected_component not in connected_components:
            connected_components.append(connected_component)
    return connected_components


def find_integer_in_sets(integer, set_list):
    for index, integer_set in enumerate(set_list):
        if integer in integer_set:
            return index  

def deledge(graph, connected_components):
    for (u, v) in graph.edges:
        # print(find_integer_in_sets(u, connected_components1))
        if v not in connected_components[find_integer_in_sets(u, connected_components)]:
            graph.remove_edge(u, v)

def find_max_set_length(set_list):
    max_length = 0
    for integer_set in set_list:
        set_length = len(integer_set)
        if set_length > max_length:
            max_length = set_length

    return max_length

def MCC(G1, G2):
    connected_components1 = find_connected_components(G1)
    connected_components2 = find_connected_components(G2)
    while connected_components1 != connected_components2:
        deledge(G2, connected_components1)
        connected_components2 = find_connected_components(G2)
        deledge(G1, connected_components2)
        connected_components1 = find_connected_components(G1)
    return find_max_set_length(connected_components1)

def delnode(G1,G2, remove_nodes):
    D1 = dict(G1.degree())
    D2 = dict(G2.degree())

    max_degrees = []
    for node in D1.keys():
        degree1 = D1[node]
        degree2 = D2[node]
        max_degrees.append((node, degree1+degree2))
        #max_degrees.append((node, max(degree1,degree2)))
    # print(max_degrees)
    # D=max(max_degrees, key=lambda t:t[1])
    max_degree = max(max_degrees, key=lambda t: t[1])[1]
    max_degree_nodes = [t[0] for t in max_degrees if t[1] == max_degree]
    # print(len(max_degree_nodes))
    random_node = random.choice(max_degree_nodes)
    remove_nodes.append(random_node)
    G1.remove_node(random_node)
    G2.remove_node(random_node)
    dN = 1

    return G1, G2, dN

def critical_number(G1, G2, N, M):
    MCCs = [1]
    ps = [1]
    dN = 0
    num = N
    G3 = G1.copy()
    G4 = G2.copy()
    lastm = M
    remove_nodes = []
    while lastm != 1:
    #while num>0:
        G5, G6, add_dN = delnode(G3, G4, remove_nodes)
        num -= 1
        m = MCC(G5, G6)
        G3 = G5.copy()
        G4 = G6.copy()
        value = m / M
        if m <= 0.4 * M and m > math.sqrt(M):
            # if m < lastm:
            dN += add_dN
                # print('dN = {}    m/M = {}'.format(dN, value))
        # elif m <= math.sqrt(M):
            # if m < lastm:
            # dN += add_dN
                # print('dN = {}    m/M = {}'.format(dN, value))
            # break
        MCCs.append(value)
        ps.append(num/N)
        lastm = m
    dN += 1

    return dN, MCCs, ps, remove_nodes
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
    
def draw_percolation_transition(fig, dataname, MCCs, ps, num1, num2, type, label):

    plt.plot(ps, MCCs, type, label=label)

    plt.title('{}_{}_{}'.format(dataname[:3], num1, num2))
    plt.xlabel('p')
    plt.ylabel('MCC')

    plt.xlim(0.7, 1.0)

    x_major_locator = MultipleLocator(0.1) 
    x_minor_locator = MultipleLocator(0.05)  

    plt.gca().xaxis.set_major_locator(x_major_locator)
    plt.gca().xaxis.set_minor_locator(x_minor_locator)

    return fig




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
    file_path = '../../../results/HDA/synthetic'
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    test_name = ['30-50','50-100','100-200','200-300','300-400','400-500']
    for data in test_name :
        score = []
        for i in tqdm(range(100)):
            adj1 = np.load("../../data/syn_%s/adj1_%s.npy"%(data,i))
            adj2 = np.load("../../data/syn_%s/adj2_%s.npy"%(data,i))
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
            dN, MCCs, ps, remove_nodes = critical_number(G1, G2, N, M_ori)
            #temp_score = (sum(MCCs)-1)/N + (N-len(remove_nodes)-1) / (M_ori * N)
            temp_score = (sum(MCCs)-1)/N
            score.append(temp_score)
                
        score_mean = np.mean(score)
        score_std = np.std(score)
        print(score_mean*100,score_std*100)
        with open('%s/test_add_result_%s.txt'%(file_path,data), 'w') as fout:
            fout.write('%.2f±%.2f,' % (score_mean * 100, score_std * 100))


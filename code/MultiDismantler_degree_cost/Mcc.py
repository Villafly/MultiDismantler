#coding=utf-8
import networkx as nx
def deledge(graph, index_map, remove_edge):
    for (u, v) in graph.edges:
        if u in index_map and v in index_map and index_map[u] == index_map[v]:
            continue
        else:
            graph.remove_edge(u,v)
            remove_edge.add((u,v))
            remove_edge.add((v,u))

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

def MCC(G1, G2, remove_edge):
    connected_components1,index_map1 = find_connected_components(G1)
    connected_components2,index_map2 = find_connected_components(G2)
    while connected_components1 != connected_components2:
        deledge(G2, index_map1,remove_edge[1])
        connected_components2, index_map2 = find_connected_components(G2)
        deledge(G1, index_map2,remove_edge[0])
        connected_components1, index_map1 = find_connected_components(G1)
    return connected_components1



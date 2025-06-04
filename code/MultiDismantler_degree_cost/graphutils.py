from typing import List
#no problem

class GraphUtil:
    def __init__(self):
        pass

    #     # Methods for deleting specified nodes  
    # def delete_node(self, adj_list_graph: List[List[int]], node: int):
    def delete_node(self, adj_list_graph, node):
        
        for j in range(2):
            for i in range(len(adj_list_graph[j][node])):
                
                neighbour = adj_list_graph[j][node][i]
                
                adj_list_graph[j][neighbour].remove(node)
            
            adj_list_graph[j][node].clear()


    # Methods to restore and add nodes
    #def recover_add_node(self, backup_completed_adj_list_graph: List[List[int]], backup_all_vex: List[bool],adj_list_graph: List[List[int]], node: int, union_set: DisjointSet):
    def recover_add_node(self, backup_completed_adj_list_graph, backup_all_vex,
                            adj_list_graph, node, union_set):
        for i in range(2):
           
            for neighbor_node in backup_completed_adj_list_graph[i][node]:
                
                if backup_all_vex[neighbor_node] :
                    
                    
                    self.add_edge(adj_list_graph[i], node, neighbor_node)
                    union_set[i].merge(node, neighbor_node)
        backup_all_vex[node] = True


    # Methods for adding edges
    def add_edge(self, adj_list_graph: List[List[int]], node0: int, node1: int):
        
        max_node = max(node0, node1)
       
        if len(adj_list_graph) - 1 < max_node:
            adj_list_graph.extend([] for _ in range(max_node - len(adj_list_graph) + 1))
            
        adj_list_graph[node0].append(node1)
        adj_list_graph[node1].append(node0)


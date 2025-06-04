# graph_struct.py

# The LinkedTable class represents a linked table structure to which elements can be dynamically added.
class LinkedTable:
    def __init__(self):
        
        self.n = 0
       
        self.ncap = 0
       
        #self.head = []
        self.head = [[]]

       

    def add_entry(self, head_id, content):
        """将内容添加到指定linked list中"""
        # If the list position to be added is greater than the current number of nodes
        if head_id >= self.n:
            # 
            if head_id + 1 > self.ncap:
                # 
                self.ncap = max(self.ncap * 2, head_id + 1)
            
                self.head.extend([[] for _ in range(head_id + 1 - self.n)])
                #
            self.n = head_id + 1

            
        self.head[head_id].append(content)
       

    def resize(self, new_n):
        """调整表格大小"""
        # If the new capacity is greater than the current capacity
        if new_n > self.ncap:
            
            self.ncap = max(self.ncap * 2, new_n)
           
            self.head.extend([[] for _ in range(new_n - len(self.head))])
            
        self.n = new_n
        
        for entry in self.head:
            if entry is not None:
                entry.clear()

       

#The GraphStruct class represents the structure of the graph, containing information about incoming edges, outgoing edges, #subgraphs, and the list of edges.
class GraphStruct:
    def __init__(self):
       
        self.out_edges = LinkedTable()  
        self.in_edges = LinkedTable()  
        self.subgraph = LinkedTable() 
        self.edge_list = []  
        self.num_nodes = 0  
        self.num_edges = 0  
        self.num_subgraph = 0  

    # The add_edge method is used to add an edge to the graph and update the associated information
    def add_edge(self, idx, x, y):
        """添加一条边"""
        
        self.out_edges.add_entry(x, (idx, y))
        
        self.in_edges.add_entry(y, (idx, x))
        
        self.num_edges += 1
        
        self.edge_list.append((x, y))
       
        assert self.num_edges == len(self.edge_list)
       
        assert self.num_edges - 1 == idx
       

    def resize(self, num_subgraph, num_nodes=0):
        self.num_nodes = num_nodes
        self.num_edges = 0
        self.edge_list = []
        self.num_subgraph = num_subgraph
        self.in_edges.resize(self.num_nodes)
        self.out_edges.resize(self.num_nodes)
        self.subgraph.resize(self.num_subgraph)

    def add_node(self,subg_id,n_idx):
        self.subgraph.add_entry(subg_id,n_idx)
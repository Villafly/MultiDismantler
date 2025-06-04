# Merge and query set classes for handling merge and query operations on collections
class DisjointSet:
    # Initialization method that creates a concatenated set containing the set size and initial rank (rank)
    def __init__(self, graphSize):
        # union_set  Used to store the parent node of each node, which at initialization is each node's own parent node
        self.union_set = [0] * graphSize
        for i in range(graphSize):
            self.union_set[i] = i
        # rank_count Used to store the rank of each set (represented by the root node), all with initial rank 1
        self.rank_count = [1] * graphSize
        # max_rank_count Record the rank of the set with the largest rank in the current union set
        self.max_rank_count = 1
        self.ccd_score = 0.0
        # Method for finding the root node with path compression optimization

    def find_root(self, node):
        
        if node != self.union_set[node]:
            
            rootNode = self.find_root(self.union_set[node])
            self.union_set[node] = rootNode 
            return rootNode
        else:
          
            return node


    def merge(self, node1, node2):
      
        node1_root = self.find_root(node1)
        node2_root = self.find_root(node2)
        maxn = 0
        
        if node1_root != node2_root:
           
            node1_rank = self.rank_count[node1_root]
            node2_rank = self.rank_count[node2_root]
            temp1 = node1_rank * (node1_rank - 1) / 2.0 + node2_rank * (node2_rank - 1) / 2.0
            
            self.ccd_score -= temp1
            temp2 = (node1_rank + node2_rank) * (node1_rank + node2_rank - 1) / 2.0
            self.ccd_score += temp2

           
            if node2_rank > node1_rank:
                self.union_set[node1_root] = node2_root  
                self.rank_count[node2_root] += self.rank_count[node1_root]  
                
                if self.rank_count[node2_root] > self.max_rank_count:
                    self.max_rank_count = self.rank_count[node2_root]
            else:
                self.union_set[node2_root] = node1_root  
                self.rank_count[node1_root] += self.rank_count[node2_root]  
                
                if self.rank_count[node1_root] > self.max_rank_count:
                    self.max_rank_count = self.rank_count[node1_root]

    def get_biggest_component_current_ratio(self) -> float:
        
        print("len.union_set:{},graphsize:{}".format(len(self.union_set),len(self.rank_count)))
        return self.max_rank_count / len(self.union_set)

    

    def get_rank(self, rootNode) -> int:
       
        return self.rank_count[rootNode]
import random
from typing import List, Tuple
from mvc_env import MvcEnv
'''
Data class:

Represents an empirical data point.
Contains information about the graph (g), the current state sequence (s_t), the next state sequence (s_prime), the action (a_t), the reward received (r_t), and a flag indicating whether the state is terminated (term_t).

LeafResult class:
Represents the result of retrieving leaf nodes from the SumTree.
Contains the leaf node's index (leaf_idx), priority value (p), and the corresponding Data object.

SumTree class:
SumTree data structure implemented for prioritizing experience playback.Intervals are divided into subintervals, each constructing a binary tree stores experiences and their priorities, allowing for efficient sampling based on priorities.
Provides methods for adding data with a given priority, updating the priority, and obtaining leaf nodes based on random values.

ReplaySample class:
Represents a batch of experiences sampled from the playback memory.
Stores lists of various components such as graph objects (g_list), state sequences (list_st), next state sequences (list_s_primes), actions (list_at), rewards (list_rt), and termination state flags (list_term).

Memory class:
Managing playback memory used in deep reinforcement learning.
Implemented prioritized experience playback using SumTree.
Stores experiences and their prioritization.
Provides methods for storing new experiences, adding experiences with N-step payoffs from the environment, sampling a batch of experiences, and updating priorities based on error.
'''

class Data:
    def __init__(self):
        self.g = None  # the graph 
        self.s_t = []  # the current state sequence
        self.s_prime = []  # the next state sequence 
        self.a_t = 0  # the action
        self.r_t = 0.0  # the reward received
        self.term_t = False  #  a flag indicating whether the state is terminated

# LeafResult class
class LeafResult:
    def __init__(self):
        
        self.leaf_idx = 0
        self.p = 0.0
        self.data = None  

# SumTree class
class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data_pointer = 0
        self.minElement = float("inf")
        self.maxElement = 0.0
        self.tree = [0.0] * (2 * capacity - 1)
        self.data = [None] * capacity

    def add(self, p: float, data: Data):
        """
        Adds data to the SumTree.

        Parameters:
        - p: Priority of the data.
        - data: The Data class to add.
        """
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_idx: int, p: float):
        """
        Updates the priority in SumTree.

        Parameters:
        - tree_idx: index of the node to update.
        - p: The updated priority.
        """
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p

        if p < self.minElement:
            self.minElement = p

        if p > self.maxElement:
            self.maxElement = p

        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v: float) -> LeafResult:
        """
        Gets the leaf nodes in the SumTree.

        Parameters:
        - v: random value for selecting leaf nodes.

        Returns:
        LeafResult object with information about the leaf nodes
        """
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        result = LeafResult()
        result.leaf_idx = leaf_idx
        result.p = self.tree[leaf_idx]
        result.data = self.data[data_idx]
        return result

# ReplaySample class
class ReplaySample:
    def __init__(self, batch_size: int):
        """
        Constructor for the ReplaySample object.

        Parameters:
        - batch_size: batch size of the sample
        """
        self.b_idx = [0] * batch_size
        self.ISWeights = [0.0] * batch_size
        self.g_list = []   
        self.list_st = []  
        self.list_s_primes = []  
        self.list_at = []  
        self.list_rt = []  
        self.list_term = []  

# Memory class
class Memory:
    def __init__(self, epsilon: float, alpha: float, beta: float, beta_increment_per_sampling: float, abs_err_upper: float, capacity: int):
        """
        Constructor for Memory objects.

        Parameters:
        - epsilon: Minor value to be used for priority updates.
        - alpha: priority sampling index.
        - beta: importance sampling index.
        - beta_increment_per_sampling: increment of the importance sampling index.
        - abs_err_upper: The upper limit of the priority.
        - capacity: Memory capacity.
        """
        self.tree = SumTree(capacity)
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.abs_err_upper = abs_err_upper

    def store(self, transition: Data):
        """
        Stores data into Memory.

        Parameters:
        - transition: the Data class to store.
        """
        max_p = self.tree.maxElement
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)

    def add(self, env: MvcEnv, n_step: int):
        """
        Add MvcEnv experience to Memory.

        Parameters:
        - env: MvcEnv object.
        - n_step: n number of steps to return.
        """
        assert env.isTerminal()
        num_steps = len(env.state_seq)
        assert num_steps > 0

        env.sum_rewards[num_steps - 1] = env.reward_seq[num_steps - 1]
        for i in range(num_steps - 1, -1, -1):
            if i < num_steps - 1:
                env.sum_rewards[i] = env.sum_rewards[i + 1] + env.reward_seq[i]

        for i in range(num_steps):
            term_t = False
            cur_r = 0.0
            s_prime = []
            if i + n_step >= num_steps:
                cur_r = env.sum_rewards[i]
                s_prime = env.action_list.copy()
                term_t = True
            else:
                cur_r = env.sum_rewards[i] - env.sum_rewards[i + n_step]
                s_prime = env.state_seq[i + n_step].copy()

            transition = Data()
            transition.g = env.graph
            transition.s_t = env.state_seq[i].copy()
            transition.a_t = env.act_seq[i]
            transition.r_t = cur_r
            transition.s_prime = s_prime.copy()
            transition.term_t = term_t

            self.store(transition)

    def sampling(self, batch_size: int) -> ReplaySample:
        """
        Sample a batch of data from Memory.

        Parameters:
        - batch_size: size of the batch to sample.

        Returns:
        ReplaySample object containing the sampled data.
        """
        result = ReplaySample(batch_size)
        total_p = self.tree.tree[0]
        pri_seg = total_p / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        min_prob = self.tree.minElement / total_p

        for i in range(batch_size):
            a = pri_seg * i
            b = pri_seg * (i + 1)
            v = random.uniform(a, b)
            leaf_result = self.tree.get_leaf(v)
            result.b_idx[i] = leaf_result.leaf_idx
            prob = leaf_result.p / total_p
            result.ISWeights[i] = (prob / min_prob) ** -self.beta
            result.g_list.append(leaf_result.data.g)
            result.list_st.append(leaf_result.data.s_t)
            result.list_s_primes.append(leaf_result.data.s_prime)
            result.list_at.append(leaf_result.data.a_t)
            result.list_rt.append(leaf_result.data.r_t)
            result.list_term.append(leaf_result.data.term_t)

        return result

    def batch_update(self, tree_idx: List[int], abs_errors: List[float]):
        """
        Batch update the priority of the data in Memory.

        Parameters:
        - tree_idx: list of node indexes to update.
        - abs_errors: The priority of the update.
        """
        for i in range(len(tree_idx)):
            abs_errors[i] += self.epsilon
            clipped_error = min(abs_errors[i], self.abs_err_upper)
            ps = clipped_error ** self.alpha
            self.tree.update(tree_idx[i], ps)

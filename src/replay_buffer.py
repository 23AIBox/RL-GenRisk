import numpy as np
import torch
import gc
from torch_geometric.data import Data, Batch

class ReplayBuffer:
    def __init__(self, max_size,n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_actions=n_actions
        self.embedding_size=64
        # self.graph_memory = [None] * self.mem_size
        self.memory_s = np.zeros((self.mem_size, self.n_actions,3))
        self.memory_a = np.zeros((self.mem_size, 1))
        self.memory_r = np.zeros((self.mem_size,1))
        self.memory_ai = np.zeros((self.mem_size, self.n_actions))
        self.memory_sa = np.zeros((self.mem_size, self.n_actions))
    def store_transition(self,  state, action, reward,action_index,sel_action):
        # graph = Data(edge_index=torch.tensor(edge_index, dtype=torch.float32))
        # graph.x_attr = torch.tensor(mu, dtype=torch.float32)
        # graph.state = torch.tensor(state, dtype=torch.float32)
        # graph.action = torch.tensor(action, dtype=torch.float32)
        # graph.reward = torch.tensor(reward, dtype=torch.float32)
        # graph.new_state = torch.tensor(state_, dtype=torch.float32)
        # graph.done = torch.tensor(done)
        # graph.action_index = torch.LongTensor(action_index)
        index = self.mem_cntr % self.mem_size
        self.memory_s[index, :] = state
        self.memory_a[index, :] = action
        self.memory_r[index, :] = reward
        self.memory_ai[index, :] = action_index
        self.memory_sa[index, :] = sel_action

        # index = self.mem_cntr % self.mem_size
        # self.graph_memory[index] = graph
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        if self.mem_cntr > self.mem_size:
            selete = [x for x in range(self.mem_size)]
            sample_index = np.random.choice(selete, size=batch_size)
        else:
            selete = [x for x in range(self.mem_cntr)]
            sample_index = np.random.choice(selete, size=batch_size)
        batch_s = self.memory_s[sample_index, :]
        batch_a = self.memory_a[sample_index, :]
        batch_r = self.memory_r[sample_index, :]
        batch_ai = self.memory_ai[sample_index, :]
        batch_sa=self.memory_sa[sample_index,:]
        return batch_s,batch_a,batch_r,batch_ai,batch_sa
    def clear(self):

        self.memory_ei = 0
        self.memory_s = 0
        self.memory_a = 0
        self.memory_r = 0
        self.memory_ai = 0

    def __len__(self):
        return min(self.mem_cntr, self.mem_size)


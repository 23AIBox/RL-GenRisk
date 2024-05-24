import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_add
from functools import partial

def change(matrix):
    leng=len(matrix)
    result0=[]
    result1=[]
    for i in range(leng):
        for j in range(leng):
            if matrix[i][j]!=0:
                result0.append(i)
                result1.append(j)
    result=[result0,result1]
    return result

class Q_Fun(nn.Module):
    def __init__(self, in_dim, hid_dim, T, ALPHA , net_old):
        super(Q_Fun, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.conv1 = GCNConv(hid_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim)
        self.T    = T
        Linear = partial(nn.Linear, bias=True)
        self.lin1 = Linear(3, hid_dim)
        self.lin2 = Linear(3*hid_dim, hid_dim)
        self.lin3 = Linear(3, hid_dim)
        self.lin4 = Linear(hid_dim, hid_dim)
        self.lin5 = Linear(hid_dim*2, hid_dim)
        self.lin6 = Linear(hid_dim, hid_dim)
        self.lin7 = Linear(hid_dim, hid_dim)
        self.lin8 = Linear(hid_dim,1)
        self.device = torch.device("cpu")
        self.net_old = torch.tensor(change(net_old)).long().to(self.device)
        self.net_old2 = torch.tensor(net_old).long().to(self.device)
        self.n_actions=self.net_old2.shape[0]
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)

    def forward(self, mu, x,action_sel, batch_flag=False,test_flag=False):

        if mu is None:
            x_in=x.clone()
            x_1=self.lin1(x)
            self.dropout = nn.Dropout(p=0.5,inplace=False)
            x_2 = self.conv1(x_1, self.net_old)
            x_2 = x_2.relu()
            self.dropout = nn.Dropout(p=0.5,inplace=False)
            x_3 = self.conv2(x_2, self.net_old)
            x_3 = x_3.relu()
            nodes_vec = self.lin2(torch.cat([x_1,x_2,x_3], dim=-1))
        else:
            nodes_vec=mu


        num_nodes = self.n_actions



        if not batch_flag:
            graph_pool2 = scatter_add(nodes_vec, action_sel, dim=-2)[0]
            number = len(action_sel) - torch.sum(action_sel)
            if test_flag:
                number=1
            graph_pool2 = (graph_pool2 ).repeat(num_nodes, 1)
            
        else:
            graph_pool2 = scatter_add(nodes_vec, action_sel, dim=-2)[:, [0]]
            number = num_nodes - torch.sum(action_sel, 1, keepdim=True).unsqueeze(1)
            if test_flag:
                number=1
            graph_pool2 = (graph_pool2 ).repeat(1, num_nodes, 1)
        Cat = torch.cat((self.lin6(graph_pool2), nodes_vec), dim=-1)
        return self.lin8(F.relu(self.lin5(F.relu(Cat)))).squeeze(),nodes_vec


import torch as th 
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import EvolveGCNO

class GNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(GNNAgent, self).__init__()
        print("Using GNN agent")
        self.args = args
        self.N = self.args.n_agents 

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        # self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.rnn = EvolveGCNO(in_channels=args.rnn_hidden_dim) 
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions) 
        self.edge_index = self.build_edge_index("full") 

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def build_edge_index(self, type):
        if type == "line": 
            edges = [[i, i + 1] for i in range(self.N - 1)]  # # arrange agents in a line 
        elif type == "full": 
            edges = [[(j, i + j + 1) for i in range(self.N - j - 1)] for j in range(self.N - 1)]
            edges = [e for l in edges for e in l] 
        elif type == 'cycle':    # arrange agents in a circle
            edges = [(i, i + 1) for i in range(self.N - 1)] + [(self.N - 1, 0)] 
        elif type == 'star':     # arrange all agents in a star around agent 0
            edges = [(0, i + 1) for i in range(self.N - 1)] 
        edge_index = th.tensor(edges).T.cuda() # # arrange agents in a line 
        return edge_index
    
    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        # h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        # h = self.rnn(x, h_in)
        h = self.rnn(x, self.edge_index)
        q = self.fc2(h)
        return q, h

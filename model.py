import torch
from torch import nn
from torch_geometric.nn import GatedGraphConv
import torch.nn.functional as F   
import math
    
class TA_GNN(nn.Module):
    def __init__(self, n_node, embed_dim=100) -> None:
        super().__init__()

        self.item_embedding = nn.Embedding(num_embeddings=n_node, embedding_dim=embed_dim)
        self.hidden_size = embed_dim
        self.Conv = GatedGraphConv(self.hidden_size, num_layers=1)
        self.q = nn.Linear(self.hidden_size, 1)
        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_4 = nn.Linear(3 * self.hidden_size, self.hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, data):
        x, edge_index, batch = data.x - 1, data.edge_index, data.batch
        x = self.item_embedding(x).squeeze()
        sections = torch.bincount(batch.cpu())
        
        # GatedGraphConv layer
        x = F.relu(self.Conv(x, edge_index))

        # Attention network
        v_i = torch.split(x, tuple(sections.cpu().numpy()))
        v_n = torch.cat(tuple(nodes[-1].view(1, -1) for nodes in v_i), dim=0)
        
        # get S_Tagert
        S_s_split = tuple(self.get_scores(nodes).view(1, -1) for nodes in v_i)
        S_s = torch.cat(S_s_split, dim=0)

        # get S_global
        v_n_repeat = torch.cat(tuple(nodes[-1].view(1, -1).repeat(nodes.shape[0], 1) for nodes in v_i))
        Alpha = self.q(torch.sigmoid(self.W_2(x) + self.W_3(v_n_repeat)))
        S_g = torch.split(Alpha * x, tuple(sections.cpu().numpy()))
        S_g = torch.cat(tuple(torch.sum(nodes, dim=0).view(1, -1) for nodes in S_g), dim=0)

        S = self.W_4(torch.cat((S_s, S_g, v_n), dim=1))

        y_hat = torch.mm(S, self.item_embedding.weight.transpose(1, 0))

        return y_hat

    def get_scores(self, s):
        v_n = s[-1].view(1, -1)
        score = F.softmax(torch.mm(v_n, s.T), dim=1).view(-1, 1)
        S_s = torch.sum(score * s, dim=0)

        return S_s

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from Utils import load_data, load_graph
from Gnn import GraphAttentionLayer,SELayer


class GEBCM(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha):
        super(GEBCM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.conv0 = SELayer(num_features)
        self.conv1 = GraphAttentionLayer(num_features, hidden_size, alpha)      
        self.conv2 = GraphAttentionLayer(hidden_size, embedding_size, alpha)

    def forward(self, x, adj, M):
        x = self.conv0(x)
        h = self.conv1(x, adj, M)
        h = self.conv2(h, adj, M)
        z = F.normalize(h, p=2, dim=1)          
        A_pred = dot_product_decode(z)          
        return A_pred,z

def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred

def pretrain_gebcm(dataset):
    model = GEBCM(num_features=args.input_dim, hidden_size=args.hidden1_dim,
                  embedding_size=args.hidden2_dim,alpha=args.alpha).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    #Some porcess
    adj, adj_label = load_graph(args.name, args.k)
    adj_dense = adj.to_dense()
    adj_numpy = adj_dense.data.cpu().numpy()
    t=2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    M = torch.Tensor(M_numpy).cuda()
    adj = adj_dense.cuda()
    adj_label = adj_label.cuda()

    data = torch.Tensor(dataset.x).cuda()
    y = dataset.y

    for epoch in range(args.max_embedding_epoch):
        model.train()
        A_pred,z = model(data, adj, M)

        loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1))
        optimizer.zero_grad()       
        loss.backward()             
        optimizer.step()            

    torch.save(model.state_dict(), 'pregebcm.pkl')
    print('Save pkl file!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='pre_train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='vcperf')
    parser.add_argument('--max_embedding_epoch', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--hidden1_dim', default=256, type=int)
    parser.add_argument('--hidden2_dim', default=32, type=int)
    parser.add_argument('--weight_decay', type=int, default=5e-3)
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.k = None
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")
    dataset = load_data(args.name)

    args.input_dim = dataset.x.shape[1]
    pretrain_gebcm(dataset)
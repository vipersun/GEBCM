import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.preprocessing import normalize
from Utils import load_data, load_graph, visual, visual_umap
from Evaluation import compactness,separation,DVI,cluster_prc_f1,translateoutliner
from PreGebcm import GEBCM
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
import sys
import math


def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

class Self_GEBCM(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha, num_clusters, v=1):
        super(Self_GEBCM, self).__init__()
        self.num_clusters = num_clusters
        self.v = v
        # get pretrain model
        self.pre_gebcm = GEBCM(num_features, hidden_size, embedding_size, alpha)
        # load thr pre-trained GAT model
        self.pre_gebcm.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(num_clusters, embedding_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x, adj, M):
        # embedding: Z
        A_pred, z = self.pre_gebcm(x, adj, M)
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        # 噪声
        # q_n = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / (self.v+float(np.random.normal(loc=0,scale=0.1,size=1))))
        # q_n = q_n.pow((self.v + 1.0) / 2.0)
        # q_n = (q_n.t() / torch.sum(q_n, 1)).t()
        q_n = q + float(np.random.normal(loc=0,scale=0.1,size=1))
        return A_pred, z, q, q_n

def gebcm(dataset):
    model = Self_GEBCM(num_features=args.input_dim, hidden_size=args.hidden1_dim,
                  embedding_size=args.hidden2_dim, alpha=args.alpha, num_clusters=args.n_clusters).to(device)
    # print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Some porcess
    adj, adj_label = load_graph(args.name, args.k)
    adj_dense = adj.to_dense()
    adj_numpy = adj_dense.data.cpu().numpy()
    t = 2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    M = torch.Tensor(M_numpy).cuda()
    adj = adj_dense.cuda()
    adj_label = adj_label.cuda()
    # cluster parameter initiate
    data = torch.Tensor(dataset.x).cuda()
    y = dataset.y
    with torch.no_grad():
        _, z = model.pre_gebcm(data, adj, M)

    # kmeans
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    for i in range(args.max_iteration_epoch):
        y_pred = kmeans.fit_predict(z.data.cpu().numpy())
        model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
        for epoch in range(args.max_distribution_epoch):
            model.train()
            if epoch % args.update_interval == 0:
                # update_interval
                A_pred, z, tmp_q , q_n= model(data, adj, M)
                tmp_q = tmp_q.data
                p = target_distribution(tmp_q)
                res1 = tmp_q.cpu().numpy().argmax(1)  # Q
                res3 = p.data.cpu().numpy().argmax(1)  # P

            A_pred, z, q, q_n= model(data, adj, M)
            kl_loss = F.kl_div(q.log(), p, reduction='batchmean')                                   
            re_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1))
            loss = args.r * kl_loss + re_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # sort
    res1 = list(res1)
    n_class = len(set(res1))
    res1[:] = [y - min(res1) for y in res1]     
    while(max(res1) != n_class - 1):      
        for k in range(n_class):
            if k in res1:
                continue
            res1[:] = [y-1 if y >= k else y for y in res1]    
        n_class = len(set(res1))
    return res1,z

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='vcperf')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--max_distribution_epoch', type=int, default=30)
    parser.add_argument('--max_iteration_epoch', type=int, default=10)
    parser.add_argument('--update_interval', default=5, type=int)
    parser.add_argument('--num_of_module', default=10, type=int)
    # parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--r', default=10, type=int)
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
    fun_num = dataset.x.shape[0]
    args.pretrain_path = 'pregebcm.pkl'
    args.n_clusters = fun_num // args.num_of_module

    y = dataset.y
    resq, z = gebcm(dataset)
    resq_later = translateoutliner(list(resq))
    prc,f1 = cluster_prc_f1(y, resq_later)
    print('f1:',f1)

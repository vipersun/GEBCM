import torch
import torch.nn.functional as F
import torch.nn as nn

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.kaiming_uniform_(self.W.data, a=0,mode='fan_out',nonlinearity='leaky_relu') 

        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.kaiming_uniform_(self.a_self.data, a=0,mode='fan_out',nonlinearity='leaky_relu') 

        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.kaiming_uniform_(self.a_neighs.data, a=0,mode='fan_out',nonlinearity='leaky_relu') 

        self.leakyrelu = nn.LeakyReLU(self.alpha)       #激活函数

    def forward(self, input, adj, M, concat=True):    # input [N, in_features]
        h = torch.mm(input, self.W)                   # shape [N, out_features] ,h是对输入特征降维后的矩阵，对应论文里的h'，N是点的个数
        #前馈神经网络
        attn_for_self = torch.mm(h,self.a_self)       #(N,1)
        attn_for_neighs = torch.mm(h,self.a_neighs)   #(N,1)
        attn_dense = attn_for_self + torch.transpose(attn_for_neighs,0,1)  #(N,N) 论文中Fig1左图
        attn_dense = torch.mul(attn_dense,M)                # [N, N]*[N, N]=>[N, N]
        attn_dense = self.leakyrelu(attn_dense)             #(N,N)

        #掩码（邻接矩阵掩码）
        zero_vec = -9e15*torch.ones_like(adj)               #(N,N)
        adj = torch.where(adj > 0, attn_dense, zero_vec)
        attention = F.softmax(adj, dim=1)                   # 对每一行的样本所有邻居softmax,归一化
        h_prime = torch.matmul(attention,h)                 # N, output_feat

        if concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
            return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SELayer(nn.Module):
    """
    Input:  Functions attribute Tensor, Tensor.size() = (number of funciotns, attribute of function)
    Output: Functions attribute Tensor, Tensor.size() = (number of funciotns, attribute of function with weight)

    Process:1. Regard all of the function as a batch;
            2. Regard the dim of attribute as channels;
            3. Reagrd the number of function as height;
            4. Set width to 1;
            5. Execute SEblock.

    """

    def __init__(self, channel, reduction=32):
        super(SELayer, self).__init__()
        channel = channel
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        # self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        xt = x.transpose(0,1)               # transpose
        xt_3 = torch.unsqueeze(xt,0)        # unsqueeze batch
        xt_4 = torch.unsqueeze(xt_3,3)      # unsqueeze width
        batch, channels, height, width = xt_4.size()
        y_avg = self.squeeze(xt_4).view(batch,channels)
        y = self.excitation(y_avg).view(batch,channels)
        return x * y.expand_as(x)

 

        


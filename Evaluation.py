import torch
import numpy as np
from munkres import Munkres
from sklearn import metrics

# similar to https://github.com/karenlatong/AGC-master
def cluster_prc_f1(y_true, y_pred):
    y_pred = list(y_pred)
    n_class = len(set(y_pred))
    y_pred[:] = [y - min(y_pred) for y in y_pred]
    while(max(y_pred) != n_class - 1):      # 中间有序号丢失
        for k in range(n_class):
            if k in y_pred:
                continue
            for pred in y_pred:
                if pred > k:
                    y_pred[y_pred.index(pred)] = pred - 1

    # best mapping between true_label and predict label
    l1 = list(set(y_true))
    numclass1 = len(l1)         # real
    l2 = list(set(y_pred))
    numclass2 = len(l2)         # pred

    ind = 0
    if numclass1 > numclass2:
        for i in l1:
            ind += 1
            if i in l2:
                continue
            else:
                y_pred[ind] = i
        l2 = list(set(y_pred))
        numclass2 = len(l2)
        cost = np.zeros((numclass1, numclass2), dtype=int)
    else:
        numclass_max = max([numclass1, numclass2])
        cost = np.zeros((numclass_max, numclass_max), dtype=int)
    
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    # acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    precision_weight = metrics.precision_score(y_true, new_predict, average='macro')
    # recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    # precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    # recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    f1_weight = metrics.f1_score(y_true, new_predict, average='weighted')
    # return acc, f1_micro
    return precision_weight, f1_micro

def compactness(y_pred, z):
    y_pred = list(y_pred)
    n_class = len(set(y_pred))
    x,dim_f = z.shape
    y_pred[:] = [y - min(y_pred) for y in y_pred]
    n = 0
    while(max(y_pred) != n_class - 1):      # 中间有序号丢失
        for k in range(n_class):
            if k in y_pred:
                continue
            for pred in y_pred:
                if pred > k:
                    y_pred[y_pred.index(pred)] = pred - 1
            n += 1
    n_class -= n
    if x != len(y_pred):
        print("Input error!")
        return -1
    cp_n = 0 
    for i in range(n_class):
        cp = 0
        clusters_ind = [j for j,y in enumerate(y_pred) if y == i]
        C = torch.stack([z[ind] for ind in clusters_ind],0)
        CM = torch.mean(C,dim=0)        # 簇中心      
        for ind in clusters_ind:
            cp += torch.sum(torch.pow(CM - z[ind],2)).sqrt()
        cp = cp/len(clusters_ind)
        cp_n += cp
    return float(format((cp_n / n_class),"0.6f"))#.item()


def separation(y_pred, z):
    y_pred = list(y_pred)
    n_class = len(set(y_pred))
    x,dim_f = z.shape
    y_pred[:] = [y - min(y_pred) for y in y_pred]
    n = 0
    while(max(y_pred) != n_class - 1):      # 中间有序号丢失
        for k in range(n_class):
            if k in y_pred:
                continue
            for pred in y_pred:
                if pred > k:
                    y_pred[y_pred.index(pred)] = pred - 1
            n += 1
    n_class -= n
    if x != len(y_pred) or n_class <=1:
        print("Input error!")
        return -1
    first = 0
    sp_n = 0 
    CM_N = torch.Tensor(1,dim_f).cuda()
    for i in range(n_class):
        clusters_ind = [j for j,y in enumerate(y_pred) if y == i]
        C = torch.stack([z[ind] for ind in clusters_ind],0)
        CM = torch.mean(C,dim=0).unsqueeze(0)        
        if first == 0:
            CM_N = CM 
            first = 1
            continue
        CM_N = torch.cat((CM_N,CM),dim=0)           
    for j in range(n_class):
        sp = 0
        for l in range(n_class):
            if l <= j:
                continue
            sp += torch.sum(torch.pow(CM_N[j]-CM_N[l],2)).sqrt()
        sp_n += sp
    return float(format((2*sp_n/n_class/(n_class-1)),"0.6f"))#.item()

def DVI(y_pred, z):
    y_pred = list(y_pred)
    n_class = len(set(y_pred))
    x,dim_f = z.shape
    y_pred[:] = [y - min(y_pred) for y in y_pred]
    n = 0
    while(max(y_pred) != n_class - 1):      # 中间有序号丢失
        for k in range(n_class):
            if k in y_pred:
                continue
            for pred in y_pred:
                if pred > k:
                    y_pred[y_pred.index(pred)] = pred - 1
            n += 1
    n_class -= n
    if x != len(y_pred) or n_class <=1:
        print("Input error!")
        return -1
    min_dis = 0
    max_dis = 0
    for i in range(x):
        for j in range(x):
            if j <= i:
                continue
            if y_pred[i] != y_pred[j]:
                dis = torch.sum(torch.pow(z[i]-z[j],2)).sqrt()
                if dis <= min_dis or min_dis == 0:
                    min_dis = dis
            else:
                dis = torch.sum(torch.pow(z[i]-z[j],2)).sqrt()
                if dis >= max_dis:
                    max_dis = dis         
    return float(format((min_dis / max_dis),"0.6f"))#.item()

def deloutliner(n_list):
    l = len(n_list)
    r = l/2
    min_k = r *0.8
    out_list = list()
    for i in n_list:
        real_k = 0
        for j in n_list:
            if abs(j-i) < r:
                real_k += 1
        if real_k < min_k:
            out_list.append(i)
    return out_list

def translateoutliner(label_ori):
    label = list()
    for val in label_ori:
        label.append(int(val))
    label_rel = label
    n = len(list(set(label)))
    n_clusters_dict = dict()
    mid_nodes = list() 
    for i in range(n):
        n_clusters_dict[i] = [j for j, x in enumerate(label) if x is i]
        # print(n_clusters_dict[i])
        mid_nodes.append(sum(n_clusters_dict[i])/len(n_clusters_dict[i]))
    for i in range(n):
        if deloutliner(n_clusters_dict[i]) == []:
            continue
        for out in deloutliner(n_clusters_dict[i]):
            # print(out)
            label[out] = mid_nodes.index(min(mid_nodes, key=lambda x: abs(x - out))) 
    if label == label_rel:
        return label
    else:
        return translateoutliner(label)

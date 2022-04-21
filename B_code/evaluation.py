import numpy as np
from munkres import Munkres, print_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment as linear
from sklearn import metrics
from scipy.special import comb 

def best_map(L1,L2):
    #L1 should be the groundtruth labels and L2 should be the clustering labels we got
    try:
        Label1 = np.unique(L1)
        nClass1 = len(Label1)
        Label2 = np.unique(L2)
        nClass2 = len(Label2)
        nClass = np.maximum(nClass1,nClass2)
        G = np.zeros((nClass,nClass))
        for i in range(nClass1):
            ind_cla1 = L1 == Label1[i]
            ind_cla1 = ind_cla1.astype(float)
            for j in range(nClass2):
                ind_cla2 = L2 == Label2[j]
                ind_cla2 = ind_cla2.astype(float)
                G[i,j] = np.sum(ind_cla2 * ind_cla1)
        m = Munkres()
        index = m.compute(-G.T)
        index = np.array(index)
        c = index[:,1]
        newL2 = np.zeros(L2.shape)
        for i in range(nClass2):
            newL2[L2 == Label2[i]] = Label1[c[i]]
        return newL2
    except:
        newL2 = np.zeros(L2.shape)
        return newL2  

def eva(y_true, y_pred, epoch=0):
    acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, best_map(y_true, y_pred))
    ari = ari_score(y_true, y_pred)
    ri = ri_score(y_true, y_pred)
    pur = purity_score(y_true, best_map(y_true, y_pred))
    return acc,nmi,pur, ari,ri,  f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro 

def ri_score(clusters, classes):
    try:
        tp_plus_fp = comb(np.bincount(clusters), 2).sum()
        tp_plus_fn = comb(np.bincount(classes), 2).sum()
        A = np.c_[(clusters, classes)]
        tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
                for i in set(clusters))
        fp = tp_plus_fp - tp
        fn = tp_plus_fn - tp
        tn = comb(len(A), 2) - tp - fp - fn
        return (tp + tn) / (tp + fp + fn + tn)     
    except:
        return 0

def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

def err_rate(gt_s, s):
    c_x = best_map(gt_s,s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    NMI = metrics.normalized_mutual_info_score(gt_s, c_x)
    purity = 0
    N = gt_s.shape[0]
    Label1 = np.unique(gt_s)
    nClass1 = len(Label1)
    Label2 = np.unique(c_x)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    for label in Label2:
        tempc = [i for i in range(N) if s[i] == label]
        hist,bin_edges = np.histogram(gt_s[tempc],Label1)
        purity += max([np.max(hist),len(tempc)-np.sum(hist)])
    purity /= N
    return 1-missrate,NMI,purity

def cluster_acc(y_true, y_pred):
    try:
        y_true = y_true - np.min(y_true)
        l1 = list(set(y_true))
        numclass1 = len(l1)
        l2 = list(set(y_pred))
        numclass2 = len(l2)
        ind = 0
        if numclass1 != numclass2:
            for i in l1:
                if i in l2:
                    pass
                else:
                    y_pred[ind] = i
                    ind += 1
        l2 = list(set(y_pred))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            # print('numclass1 != numclass2')
            return 0, 0, 0, 0, 0, 0, 0
        cost = np.zeros((numclass1, numclass2), dtype=int)
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
        acc = metrics.accuracy_score(y_true, new_predict)
        f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
        precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
        recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
        f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
        precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
        recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro
    except:
        return 0, 0, 0, 0, 0, 0, 0
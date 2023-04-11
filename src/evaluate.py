from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import math
import matplotlib.pyplot as plt
import numpy as np

def hit_ratio_score(y_true,y_score,k):
    order = np.argsort(y_score)[::-1]
    y_true_k = np.take(y_true, order[:k])
    return np.sum(y_true_k), np.sum(y_true)

def precision_score(y_true,y_score,k):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    return np.sum(y_true) / k

def dcg_score(y_true, y_score, k):
    order = np.argsort(y_score)[::-1]
    #print('****************** top ', k, ' ******************')
    y_true = np.take(y_true, order[:k])
    #print(y_true)
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def mrr_score(y_true, y_score, k):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    #denominator may be zero
    return np.sum(rr_score) / np.sum(y_true)

def f1_score(precision, recall):
    return float(format(2 * precision * recall / (precision + recall), '.4f'))

def score(user, label, scores, topk):
    # topk = [1,5,10,15,20,25,30,40,50]
    aucs = []
    ndcgs = [[] for i in topk]
    precisions = [[] for i in topk]
    #mrrs = [[] for i in topk]
    hits = [0 for i in topk]
    hits_and_misses = [0 for i in topk]

    user_score = {}
    user_label = {}
    for i in range(len(user)):
        u = user[i]
        if u not in user_score:
            user_label[u] = [label[i]]
            user_score[u] = [scores[i]]
        else:
            user_label[u].append(label[i])
            user_score[u].append(scores[i])

    for u in set(user):
        y_true = user_label[u]
        y_score = user_score[u]

        auc = roc_auc_score(y_true, y_score)
        aucs.append(auc)
        # topk = [1,5,10,15,20,25,30,40,50]
        for i in range(len(topk)):
            k = topk[i]
            ndcgs[i].append(ndcg_score(y_true, y_score, k))
            precisions[i].append(precision_score(y_true, y_score, k))
            #mrrs[i].append(mrr_score(y_true, y_score, i+1))
            hiti, hit_and_missi = hit_ratio_score(y_true, y_score, k)
            hits[i] += hiti
            hits_and_misses[i] += hit_and_missi

    ndcgs = [float(format(np.mean(i),'.4f')) for i in ndcgs]
    # mrrs = [float(format(np.mean(i), '.4f')) for i in mrrs]
    precisions = [float(format(np.mean(i),'.4f')) for i in precisions]
    hit_ratio = [float(format(hits[i]/hits_and_misses[i],'.4f')) for i in range(len(topk))]
    # f1_score = [f1_score(precisions[i], hit_ratio[i]) for i in range(len(topk))]


    return float(format(np.mean(aucs),'.4f')), ndcgs, precisions, hit_ratio
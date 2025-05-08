from sklearn.metrics import ndcg_score
import numpy as np

def compute_ndcg(pred_ranks, k):
    y_true = np.array([[1 if i==pred_rank else 0 for i in range(min(20,k))] for pred_rank in pred_ranks])
    y_score = np.array([[20-i for i in range(min(20,k))] for pred_rank in pred_ranks])
    ndcg = ndcg_score(y_true, y_score, k=k)
    return ndcg

def compute_hr(pred_ranks, k):
    hr = np.mean([1 if pred_rank<k else 0 for pred_rank in pred_ranks])
    return hr
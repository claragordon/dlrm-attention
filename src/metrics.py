import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

def top_decile_calibration_ratio(y_true: np.ndarray, p: np.ndarray, q: float = 0.90) -> float:
    thresh = np.quantile(p, q)
    mask = p >= thresh
    # avoid div-by-zero if mask has no positives (rare)
    denom = y_true[mask].mean() if mask.any() else 0.0
    numer = p[mask].mean() if mask.any() else 0.0
    if denom <= 1e-12:
        return float("inf") if numer > 0 else 1.0
    return float(numer / denom)

def compute_metrics(y_true: np.ndarray, logits: np.ndarray):
    # stable sigmoid
    p = 1.0 / (1.0 + np.exp(-logits))
    # logloss expects probs
    ll = log_loss(y_true, p, labels=[0,1])
    auc = roc_auc_score(y_true, p)
    auprc = average_precision_score(y_true, p)
    top10 = top_decile_calibration_ratio(y_true, p, 0.90)
    top1 = top_decile_calibration_ratio(y_true, p, 0.99)
    return {"logloss": ll, "auc": auc, "auprc": auprc, "calib_ratio_top10": top10, "calib_ratio_top1": top1}
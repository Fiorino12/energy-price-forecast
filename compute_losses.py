import numpy as np
"""
Set of functions used to compute the metrics for evaluating model performance
- Pinball scores
- Winkler scores
- Delta coverage
- CRPS
"""

def compute_pinball_scores(y_true, pred_quantiles, quantiles_levels):
    """
    Utility function to compute the pinball score on the test results
    return: pinball scores computed for each quantile level and each step in the pred horizon
    """
    score = []
    for i, q in enumerate(quantiles_levels):
        error = np.subtract(y_true, pred_quantiles[:, :, i])
        loss_q = np.maximum(q * error, (q - 1) * error)
        score.append(np.expand_dims(loss_q,-1))
    score = np.mean(np.concatenate(score, axis=-1), axis=0)
    return score

def compute_winkler_scores(y_true, pred_quantiles, quantiles_levels):
    """
    Utility function to compute the winkler score on the test results
    return: winkler scores computed for each quantile level and each step in the pred horizon
    """
    score = []
    n = len(quantiles_levels)
    for i, q in enumerate(quantiles_levels[:n//2]):
        delta = np.subtract(pred_quantiles[:, :, -i-1], pred_quantiles[:, :, i])
        delta_u = 1/q*np.subtract(y_true,pred_quantiles[:, :, -i-1])*(y_true>pred_quantiles[:, :, -i-1])
        delta_l = 1/q*np.subtract(pred_quantiles[:, :, i], y_true)*(y_true<pred_quantiles[:, :, i])
        loss_q = delta + delta_u + delta_l
        score.append(np.expand_dims(loss_q,-1))
    score = np.mean(np.concatenate(score, axis=-1), axis=0)
    return score

def compute_delta_coverage(y_true, pred_quantiles, quantiles_levels):
    """
    Utility function to compute the delta coverage on the test results
    return: delta coverage metric
    """
    score = []
    n = len(quantiles_levels)
    for i, q in enumerate(quantiles_levels[:n//2]):
        hit_perc = np.mean(np.mean((y_true>pred_quantiles[:, :, i])*(y_true<pred_quantiles[:, :, -i-1])))
        score.append(np.abs(hit_perc -(-quantiles_levels[i] +  quantiles_levels[-i-1])))
    alpha_max = quantiles_levels[-1]-quantiles_levels[0]
    alpha_min = quantiles_levels[n//2 +1] -  quantiles_levels[n//2-1]
    score = np.sum(np.array(score))/(alpha_max - alpha_min )
    return score

def compute_CRPS(y_true, pred_quantiles, quantiles_levels):
    """
    Utility function to compute the delta coverage on the test results
    return: CRPS metric
    """
    score = []

    pin_err = np.mean(np.abs(np.subtract(np.expand_dims(y_true, axis=2), pred_quantiles)), axis = 2)

    for i,q in enumerate(quantiles_levels):
        score.append( np.mean(np.abs(np.subtract(np.expand_dims(pred_quantiles[:,:,i], axis = 2), pred_quantiles)), axis = 2) )
    score = np.mean(np.array(score), axis= 0)/2


    return np.mean(pin_err-score)
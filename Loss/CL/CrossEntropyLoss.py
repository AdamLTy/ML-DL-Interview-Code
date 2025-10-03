import numpy as np


def CrossEntropyLoss(logits, labels):
    '''
    logits: shape (N, N)
    labels: shape (N, )
    '''
    N = logits.shape[0]
    max_logits = np.max(logits, axis = 1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    softmax = exp_logits / np.sum(exp_logits, axis = 1, keepdims=True)

    log_probs = -np.log(softmax[np.range[N], labels])

    return np.mean(log_probs)


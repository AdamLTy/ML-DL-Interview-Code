import numpy as np


def BCE_Loss(score, y, epsilon=1e-15):
    y_hat = 1 / (1 + np.exp(-score))
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

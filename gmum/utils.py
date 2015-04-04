import numpy as np
from scipy import linalg as la
import scipy.optimize
from sklearn.base import BaseEstimator, clone

def BAC(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean([float(sum(y_true[y_true==label] == y_pred[y_true==label])) / sum(y_true==label) for label in set(y_true)])


class SphereConstrained(BaseEstimator):
    def __init__(self, objective):
        self.objective = objective

    def fit(self, x0, x1):
        self.objective.fit(x0, x1)

    def __str__(self):
        return '{} soft constrained to the unit sphere'.format(self.objective)

    def value(self, v):
        w = v.reshape(-1, self.objective.k)
        return self.objective.value(v) - la.norm(w.T.dot(w) - np.eye(self.objective.k)) ** 2

    def derivative(self, v):
        w = v.reshape(-1, self.objective.k)
        return (self.objective.derivative(v).reshape(-1, self.objective.k) - (4 * w.dot(w.T.dot(w)) - 4 * w)).reshape(
            -1)


class Negated(BaseEstimator):
    def __init__(self, objective):
        self.objective = objective

    def fit(self, x0, x1):
        self.objective.fit(x0, x1)

    def __str__(self):
        return 'Negated {}'.format(self.objective)

    def value(self, v):
        return -self.objective.value(v)

    def derivative(self, v):
        return -self.objective.derivative(v)


def maximize_entropy(x0, x1, v0, method, regularize, objective):    
    objective = clone(objective)
    objective.fit(x0, x1)
    if regularize:
        obj = Negated(SphereConstrained(objective))
    else:
        obj = Negated(objective)

    return scipy.optimize.minimize(obj.value, v0, method=method, jac=obj.derivative)

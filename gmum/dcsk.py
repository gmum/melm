import numpy as np
from scipy import linalg as la
from sklearn.base import BaseEstimator
from sklearn.covariance import LedoitWolf, EmpiricalCovariance

class DCS_kd(BaseEstimator):

    def __init__(self, k=2, gamma=1.0, covariance_estimator='ledoit-wolf'):
        self.k = float(k)
        self.gamma = gamma
        self.covariance_estimator = covariance_estimator

        if covariance_estimator == 'empirical':
            self.cov = EmpiricalCovariance(store_precision=False)
        elif covariance_estimator == 'ledoit-wolf':
            self.cov = LedoitWolf(store_precision=False)
        else:
            raise NotImplementedError('%s is not implemented' % covariance_estimator)

        self.x0 = None
        self.x1 = None

    def fit(self, x, y):
        self.x0 = x[y == min(y)]
        self.x1 = x[y == max(y)]

    def __str__(self):
        return 'Analytical Cauchy-Schwarz Divergence in {}-d'.format(self.k)

    def value(self, v):
        # We need matrix, not vector
        v = v.reshape(-1, self.k)

        ipx0 = self._ipx(self.x0, self.x0, v)
        ipx1 = self._ipx(self.x1, self.x1, v)
        ipx2 = self._ipx(self.x0, self.x1, v)

        return np.log(ipx0) + np.log(ipx1) - 2 * np.log(ipx2)

    def derivative(self, v):
        # We need matrix, not vector
        v = v.reshape(-1, self.k)

        ret = (self._d_ipx(self.x0, self.x0, v) / self._ipx(self.x0, self.x0, v)
               + self._d_ipx(self.x1, self.x1, v) / self._ipx(self.x1, self.x1, v)
               - 2 * self._d_ipx(self.x0, self.x1, v) / self._ipx(self.x0, self.x1, v))

        return ret.reshape(-1)

    def _H(self, X0, X1):
        n = (4.0 / (self.k + 2)) ** (2.0 / (self.k + 4))
        p = (-2.0 / (self.k + 4))
        return n * (X0.shape[0] ** p * self.cov.fit(X0).covariance_ + X1.shape[0] ** p * self.cov.fit(X1).covariance_)

    def _f1(self, X0, X1, v):
        Hxy = self.gamma * self.gamma * self._H(X0, X1)
        vHv = v.T.dot(Hxy).dot(v)
        # return 1.0 / np.sqrt(la.det(vHv))
        return 1.0 / (X0.shape[0] * X1.shape[0] * np.sqrt(la.det(vHv)) * (2 * np.pi) ** (self.k / 2))

    def _g1(self, X0, X1, v):
        Hxy = self.gamma * self.gamma * self._H(X0, X1)
        vHv = v.T.dot(Hxy).dot(v)
        return - self._f1(X0, X1, v) * Hxy.dot(v).dot(la.inv(vHv))

    def _f2(self, X0, X1, v):
        Hxy = self.gamma * self.gamma * self._H(X0, X1)
        vHv = v.T.dot(Hxy).dot(v)
        vHv_inv = la.inv(vHv)

        vx0 = X0.dot(v)
        vx1 = X1.dot(v)
        vx0c = vx0.dot(vHv_inv)
        vx1c = vx1.dot(vHv_inv)

        ret = 0.0
        for i in range(X0.shape[0]):
            ret += np.exp(-0.5 * ((vx0c[i] - vx1c) * (vx0[i] - vx1)).sum(axis=1)).sum()
        return ret

    def _g2(self, X0, X1, v):
        Hxy = self.gamma * self.gamma * self._H(X0, X1)
        vHv = v.T.dot(Hxy).dot(v)
        vHv_inv = la.inv(vHv)  # k x k

        vx0 = X0.dot(v)
        vx1 = X1.dot(v)
        vx0c = vx0.dot(vHv_inv)
        vx1c = vx1.dot(vHv_inv)

        eye = np.eye(v.shape[0])
        right_expr = (eye - Hxy.dot(v).dot(vHv_inv).dot(v.T))  # d x d

        d = v.shape[0]
        k = int(self.k)
        ret = 0.0
        for i in range(X0.shape[0]):
            f2_vals = np.exp(-0.5 * ((vx0c[i] - vx1c) * (vx0[i] - vx1)).sum(axis=1)).reshape(-1, 1)
            ws = (X0[i] - X1).reshape(X1.shape[0], d, 1)
            vxdiffs = (- f2_vals * (vx0[i] - vx1)).reshape(X1.shape[0], 1, k)
            ret += np.tensordot(ws, vxdiffs, ([0, 2], [0, 1]))

        return right_expr.dot(ret).dot(vHv_inv)

    def _ipx(self, X0, X1, v):
        return self._f1(X0, X1, v) * self._f2(X0, X1, v)

    def _d_ipx(self, X0, X1, v):
        return self._f1(X0, X1, v) * self._g2(X0, X1, v) + self._f2(X0, X1, v) * self._g1(X0, X1, v)


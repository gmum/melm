from sklearn import cross_validation
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.externals.joblib.parallel import Parallel, delayed
from sklearn.neighbors import KernelDensity
from sklearn.svm import SVC
from sklearn.utils.validation import check_random_state
import math
import numpy as np
import scipy.optimize
from scipy import linalg
from scipy import linalg as la
import time


SILVERMAN_CONSTANT = (4.0 / 3.0) ** 0.2


class DCS(BaseEstimator):
    def __init__(self, gamma=1.0):
        self.gamma = gamma
        self.k = 1

        self.x0 = None
        self.x1 = None

    def fit(self, x0, x1):
        if x0.size > x1.size:
            x0, x1 = x1, x0
        self.x0 = x0
        self.x1 = x1

    def __str__(self):
        return 'Analytical Cauchy-Schwarz Divergence'

    def _V(self, X, v):
        return np.mean(np.inner(X - X.mean(axis=0), v) ** 2)

    def _H(self, X0, X1, v):
        n = SILVERMAN_CONSTANT
        p = -2.0 / 5
        return n ** 2 * (X0.shape[0] ** p * self._V(X0, v) + X1.shape[0] ** p * self._V(X1, v))

    def _ipx(self, X0, X1, v):
        Hxy = self.gamma * self.gamma * self._H(X0, X1, v)
        result = 0.0

        vx0 = np.inner(v, X0)
        vx1 = np.inner(v, X1)
        for i in range(X0.shape[0]):
            result += np.exp((- ((vx0[i] - vx1) ** 2)) / (2 * Hxy)).sum()

        return result / (X0.shape[0] * X1.shape[0] * np.sqrt(2 * np.pi * Hxy))

    def _d_V(self, X, v):
        inner_v_X = np.inner(v, X).reshape((-1, 1))
        d_H1 = (inner_v_X * X).sum(axis=0)
        d_H2 = inner_v_X.sum(axis=0)
        d_H3 = X.sum(axis=0)
        return 2 * SILVERMAN_CONSTANT ** 2 / pow(X.shape[0], 12.0 / 5.0) * ( X.shape[0] * d_H1 - d_H2 * d_H3 )

    def _d_H(self, X0, X1, v):
        return self._d_V(X0, v) + self._d_V(X1, v)

    def _d_ipx(self, X0, X1, v):
        Hxy = self.gamma * self.gamma * self._H(X0, X1, v)
        d_Hxy = self.gamma * self.gamma * self._d_H(X0, X1, v)
        result = np.zeros(X0.shape[1])

        vx0 = np.inner(v, X0)
        vx1 = np.inner(v, X1)

        for i in range(X0.shape[0]):
            cur_dot = (vx0[i] - vx1).reshape((-1, 1))
            cur_dot2 = cur_dot ** 2
            cur_left = np.exp(-cur_dot2 / (2 * Hxy))
            cur_right = (cur_dot2 / (2 * Hxy) - 0.5) * d_Hxy
            cur_right -= cur_dot * (X0[i] - X1)
            result += (cur_left * cur_right).sum(axis=0)

        return result / (X0.shape[0] * X1.shape[0] * np.sqrt(2 * np.pi * Hxy ** 3))

    def derivative(self, v):
        ret = (self._d_ipx(self.x0, self.x0, v) / self._ipx(self.x0, self.x0, v)
               + self._d_ipx(self.x1, self.x1, v) / self._ipx(self.x1, self.x1, v)
               - 2 * self._d_ipx(self.x0, self.x1, v) / self._ipx(self.x0, self.x1, v))
        return ret

    def value(self, v):
        ipx0 = self._ipx(self.x0, self.x0, v)
        ipx1 = self._ipx(self.x1, self.x1, v)
        ipx2 = self._ipx(self.x0, self.x1, v)
        ret = (np.log(ipx0) + np.log(ipx1) - 2 * np.log(ipx2))
        return ret


class DCSDist(DCS):
    def __str__(self):
        return 'Truncated approximation of Cauchy-Schwarz Divergence'

    def __init__(self, gamma=1.0, eps=0.3):
        DCS.__init__(self, gamma)
        self.eps = eps

    def _ipx(self, X0, X1, v):
        Hxy = self.gamma * self.gamma * self._H(X0, X1, v)
        vx0 = np.sort(np.inner(v, X0))
        vx1 = np.sort(np.inner(v, X1))

        t = np.sqrt(np.maximum(- Hxy * np.log(2 * self.eps ** 2 * np.pi * Hxy), 0))

        result = 0.0
        left_idx = 0
        right_idx = 0
        total_matches = 0
        for i in range(X0.shape[0]):
            while left_idx < vx1.size and vx1[left_idx] + t < vx0[i]:
                left_idx += 1
            while right_idx < vx1.size and vx1[right_idx] - t <= vx0[i]:
                right_idx += 1

            result += np.exp((- (vx0[i] - vx1[left_idx:right_idx]) ** 2) / (2 * Hxy)).sum()

        return max(result / (X0.shape[0] * X1.shape[0] * np.sqrt(2 * np.pi * Hxy)), 1e-10)

    def _d_ipx(self, X0, X1, v):
        Hxy = self.gamma * self.gamma * self._H(X0, X1, v)
        d_Hxy = self.gamma * self.gamma * self._d_H(X0, X1, v)
        result = np.zeros(X0.shape[1])

        vx0 = np.inner(v, X0)
        vx1 = np.inner(v, X1)
        t = np.sqrt(np.maximum(- Hxy * np.log(2 * self.eps ** 2 * np.pi * Hxy), 0))

        left_idx = 0
        right_idx = 0
        for i in range(X0.shape[0]):
            while left_idx < vx1.size and vx1[left_idx] + t < vx0[i]:
                left_idx += 1
            while right_idx < vx1.size and vx1[right_idx] - t <= vx0[i]:
                right_idx += 1

            cur_dot = (vx0[i] - vx1[left_idx:right_idx]).reshape((-1, 1))
            cur_dot2 = cur_dot ** 2
            cur_left = np.exp(-cur_dot2 / (2 * Hxy))
            cur_right = (cur_dot2 / (2 * Hxy) - 0.5) * d_Hxy
            cur_right -= cur_dot * (X0[i] - X1[left_idx:right_idx])
            result += (cur_left * cur_right).sum(axis=0)

        return result / (X0.shape[0] * X1.shape[0] * np.sqrt(2 * np.pi * Hxy ** 3))


class DCSBinned(DCS):
    def __init__(self, gamma=1.0, eps=0.3):
        DCS.__init__(self, gamma)
        self.eps = eps

    def __str__(self):
        return 'Binned approximation of Cauchy-Schwarz Divergence'

    def _ipx(self, X0, X1, v):
        Hxy = self.gamma * self.gamma * self._H(X0, X1, v)

        vx0 = np.sort(np.inner(v, X0))
        vx1 = np.sort(np.inner(v, X1))
        x_min = np.minimum(vx0[0], vx1[0])
        x_max = np.maximum(vx0[-1], vx1[-1])

        threshold = 1 - self.eps * np.sqrt(2 * np.pi * Hxy)
        if threshold < 1e-15:
            width = x_max - x_min + 1
        else:
            width = np.sqrt(- 2 * Hxy * np.log(threshold))
        bins = int(math.ceil((x_max - x_min) / width))
        self.last_bins = bins

        bx0 = np.histogram(vx0, bins, (x_min, x_max))[0]
        bx1 = np.histogram(vx1, bins, (x_min, x_max))[0]

        full_means = (bins - 1) * width - np.arange(2 * bins - 1) * width
        nbx0 = bx0[bx0 > 0]
        nvx0 = full_means[bx0 > 0]
        nbx1 = bx1[bx1 > 0]
        nvx1 = full_means[bx1 > 0]

        result = 0.0
        for i in range(nbx0.size):
            result += (nbx0[i] * nbx1 * np.exp((- (nvx0[i] - nvx1) ** 2) / (2 * Hxy))).sum()

        return result / (X0.shape[0] * X1.shape[0] * np.sqrt(2 * np.pi * Hxy))

    def _d_ipx(self, X0, X1, v):
        Hxy = self.gamma * self.gamma * self._H(X0, X1, v)
        d_Hxy = self.gamma * self.gamma * self._d_H(X0, X1, v)
        result = np.zeros(X0.shape[1])

        vx0 = np.inner(v, X0)
        vx1 = np.inner(v, X1)
        ix0 = np.argsort(vx0)
        ix1 = np.argsort(vx1)
        vx0 = vx0[ix0]
        vx1 = vx1[ix1]
        X0 = X0[ix0]
        X1 = X1[ix1]
        x_min = np.minimum(vx0[0], vx1[0])
        x_max = np.maximum(vx0[-1], vx1[-1])

        threshold = 1 - self.eps * np.sqrt(2 * np.pi * Hxy)
        if threshold < 1e-15:
            width = x_max - x_min + 1
        else:
            width = np.sqrt(- 2 * Hxy * np.log(threshold))
        bins = int(math.ceil((x_max - x_min) / width))
        self.last_bins = bins
        bx0, ex0 = np.histogram(vx0, bins, (x_min, x_max))
        bx1, ex1 = np.histogram(vx1, bins, (x_min, x_max))

        full_means = (bins - 1) * width - np.arange(2 * bins - 1) * width
        nbx0 = bx0[bx0 > 0]
        nvx0 = full_means[bx0 > 0]
        nbx1 = bx1[bx1 > 0]
        nvx1 = full_means[bx1 > 0]

        # mX[i] - mean X ouf of those mapped to i-th bucket
        # Computing mX is quite slow. Consider moving to C/Cython
        mX = np.zeros((bins, v.size))
        binX0 = np.minimum(np.searchsorted(ex0, vx0, 'right') - 1, bins - 1)
        binX1 = np.minimum(np.searchsorted(ex1, vx1, 'right') - 1, bins - 1)
        for i in range(X0.shape[0]):
            mX[binX0[i]] += X0[i]
        for i in range(X1.shape[0]):
            mX[binX1[i]] += X1[i]
        for i in range(bins):
            bin_size = bx0[i] + bx1[i]
            if bin_size:
                mX[i] /= bin_size
        lmX = mX[bx0 > 0, :]
        rmX = mX[bx1 > 0, :]

        for i in range(nbx0.size):
            cur_dot = nvx0[i] - nvx1
            cur_dot2 = cur_dot ** 2
            cur_left = (nbx0[i] * nbx1 * np.exp(-cur_dot2 / (2 * Hxy))).reshape((-1, 1))

            cur_right = (cur_dot2.reshape((-1, 1)) / (2 * Hxy) - 0.5) * d_Hxy.reshape((1, -1))
            cur_right -= cur_dot.reshape((-1, 1)) * (lmX[i] - rmX)
            result += (cur_left * cur_right).sum(axis=0)

        return result / (X0.shape[0] * X1.shape[0] * np.sqrt(2 * np.pi * Hxy ** 3))

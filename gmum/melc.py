from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
from sklearn.externals.joblib.parallel import Parallel, delayed
from sklearn.neighbors import KernelDensity
from sklearn.svm import SVC
from sklearn.utils.validation import check_random_state
import numpy as np

from gmum.utils import maximize_entropy
from gmum.dcs import DCS


class MELC(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):

    def __get_random_point_on_sphere(self, dim, random_state):
        point = random_state.normal(size=dim)
        point /= np.sqrt((point ** 2).sum())
        return point

    def __init__(self, method="L-BFGS-B", random_state=None, n_iters=1, n_jobs=1, verbose=0,
                 use_svc_weights=True, on_sphere=True, base_objective=DCS()):
        self.method = method
        self.random_state = random_state
        self.n_iters = n_iters
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.use_svc_weights = use_svc_weights
        self.on_sphere = on_sphere
        self.base_objective = base_objective
        self.eval_stats = list()
        self.k = 1

    def _density_classification(self, x):
        if self.kde_a.score_samples([x]) > self.kde_b.score_samples([x]):
            return self.a
        else:
            return self.b

    def _find_best_w(self, x, y, random_state):
        xa = x[y == self.a]
        xb = x[y == self.b]

        points = [self.__get_random_point_on_sphere(x.shape[1], random_state) for _ in range(self.n_iters)]
        if self.use_svc_weights:
            points.append(SVC(kernel='linear', C=1, random_state=random_state, class_weight='auto').fit(x, y).coef_[0])

        rets = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(maximize_entropy)(xa, xb, point, self.method, self.on_sphere, self.base_objective)
            for point in points)

        best_i = 0
        for i in range(len(points)):
            if self.verbose:
                print "Entropy with gamma {} starting from point[{}] = {}".format(
                    self.base_objective.gamma, i, -rets[i]["fun"])
            if rets[i]["fun"] < rets[best_i]["fun"]:
                best_i = i
        return rets[best_i]["x"]

    def fit(self, x, y):
        random_state = check_random_state(self.random_state)

        self.a = np.min(y)
        self.b = np.max(y)

        self.w = self._find_best_w(x, y, random_state)
        x_r = np.dot(self.w, x.T).T
        x1 = x_r[y == self.a]
        x2 = x_r[y == self.b]

        kA = self.base_objective.gamma * 1.06
        h1 = kA * len(x1) ** (-1.0 / 5) * np.std(x1)  # silverman's rule of the thumb
        h2 = kA * len(x2) ** (-1.0 / 5) * np.std(x2)  # silverman's rule of the thumb

        self.kde_a = KernelDensity(kernel='gaussian', bandwidth=h1).fit(x1.reshape(-1, 1))
        self.kde_b = KernelDensity(kernel='gaussian', bandwidth=h2).fit(x2.reshape(-1, 1))

        self.min_v = min(x_r)
        self.max_v = max(x_r)

        self.min_c = self._density_classification(self.min_v)
        self.max_c = self._density_classification(self.max_v)

        ytr = self.predict(x, True)

        last = ytr[0]
        self.k = 0
        for i in range(1, len(ytr)):
            if ytr[i] != last:
                self.k += 1
            last = ytr[i]

    def predict(self, xt, sort=False):
        xt_r = np.dot(self.w, xt.T).T
        if sort:
            xt_r = np.sort(xt_r)

        yp = np.zeros(xt_r.size)
        for i, _x in enumerate(xt_r):
            if _x <= self.min_v:
                yp[i] = self.min_c
            elif _x >= self.max_v:
                yp[i] = self.max_c
            else:
                yp[i] = self._density_classification(_x)

        return yp

from gmum.dcsk import DCS_kd
from gmum.utils import maximize_entropy
import numpy as np
from sklearn import decomposition
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, TransformerMixin
from sklearn.externals.joblib.parallel import Parallel, delayed
from sklearn.utils.validation import check_random_state
from gmum.classifiers import SVM, KDE, KNN

class MELM(BaseEstimator, ClassifierMixin, MetaEstimatorMixin, TransformerMixin):

    def __get_random_point_on_sphere(self, dim, random_state):

        point = random_state.normal(size=dim * self.base_objective.k)
        # TODO: add orthonormalization here

        return point

    def __init__(self, k=2, gamma=1.0, method="L-BFGS-B", random_state=None, 
                 n_starts=17, n_jobs=1, on_manifold=True, use_pca_weights=True,
                 covariance_estimator='ledoit-wolf', classifier='KDE'):

        self.method = method
        self.random_state = random_state
        self.n_starts = n_starts
        self.n_jobs = n_jobs
        self.on_manifold = on_manifold
        self.use_pca_weights = use_pca_weights
        self.k = k
        self.gamma = gamma
        self.a = None
        self.b = None
        self.w = None
        self.base_objective = None
        self.covariance_estimator = covariance_estimator
        self.classifier = classifier

    def _find_best_w(self, x, y, random_state):

        points = [self.__get_random_point_on_sphere(x.shape[1], random_state) for _ in range(self.n_starts)]
        if self.use_pca_weights:
            pca = decomposition.PCA(n_components=int(self.base_objective.k))
            pca.fit(x)
            points[-1] = pca.components_

        rets = Parallel(n_jobs=self.n_jobs)(
            delayed(maximize_entropy)(x, y, point, self.method, self.on_manifold, self.base_objective)
            for point in points)

        best_i = 0
        for i in range(len(points)):
            if rets[i]["fun"] < rets[best_i]["fun"]:
                best_i = i

        return rets[best_i]["x"].reshape(-1, self.base_objective.k).T

    def fit(self, x, y):

        if len(set(y)) > 2:
            raise NotImplementedError('Currently MELM supports only binary datasets')

        self.base_objective = DCS_kd(gamma=self.gamma, k=self.k, 
                                     covariance_estimator=self.covariance_estimator)

        if self.classifier == 'KDE':
            self.clf = KDE(gamma=self.gamma)
        elif self.classifier == 'SVM':
            self.clf = SVM()
        elif self.classifier == 'KNN':
            self.clf = KNN()
        else:
            raise NotImplementedError('%s classifier is not implemented' % self.classifier)

        random_state = check_random_state(self.random_state)

        self.a = min(y)
        self.b = max(y)

        self.classes_ = np.array([self.a, self.b])

        self.w = self._find_best_w(x, y, random_state)

        self.clf.fit(self.transform(x), y)

    def transform(self, x):
        return np.dot(self.w, x.T).T

    def predict(self, x):
        return self.clf.predict(self.transform(x))

    def predict_proba(self, x):
        return self.clf.predict_proba(self.transform(x))

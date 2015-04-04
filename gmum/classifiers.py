import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from gmum.utils import BAC
from sklearn.metrics import make_scorer
from scipy import stats
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier 


class KDE(object):

    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def fit(self, X, y):

        self.a = min(y)
        self.b = max(y)

        x1 = X[y == self.a]
        x2 = X[y == self.b]
        try:
            self.kde_a = stats.gaussian_kde(x1.T, 'silverman')
            self.kde_a.set_bandwidth(self.gamma * self.kde_a.silverman_factor())
        except:
            self.kde_a = lambda x : 0

        try:
            self.kde_b = stats.gaussian_kde(x2.T, 'silverman')
            self.kde_b.set_bandwidth(self.gamma * self.kde_b.silverman_factor())
        except:
            self.kde_b = lambda x : 0

    def predict(self, x):
        return map(lambda x : self.a if x==1 else self.b, [np.sign(self.kde_a(_x) - self.kde_b(_x)) for _x in x])

    def predict_proba(self, x):
        return [max(self.kde_a(_x), self.kde_b(_x)) / (self.kde_a(_x) + self.kde_b(_x)) for _x in x]

    def __str__(self):
        return "KDE"


class SVM(object):

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        C_range = 10.0 ** np.arange(-2, 4)
        gamma_range = 10.0 ** np.arange(-5, 4)
        param_grid = dict(gamma=gamma_range, C=C_range, class_weight=['auto'], probability=[True])
        cv = StratifiedKFold(y=y, n_folds=self.k)
        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv, scoring=make_scorer(BAC))
        grid.fit(X, y)
        self.clf = SVC(**grid.best_params_)
        self.clf.fit(X, y)
        
    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(x)

    def __str__(self):
        return "SVM RBF"

class KNN(object):

    def __init__(self, k=5):
        self.k = k
    
    def fit(self, X, y):
        self.clf = KNeighborsClassifier(n_neighbors=self.k)
        self.clf.fit(X, y)
        w = np.array([1./float(y.tolist().count(l)) for l in sorted(set(y.tolist()))])
        self.weights = max(w)  * w

    def predict(self, X):
        proba = self.clf.predict_proba(X)
        return [self.clf.classes_[r] for r in np.argmax(np.multiply(proba, self.weights), axis=1).ravel()]

    def predict_proba(self, X):
        proba = self.clf.predict_proba(X)
        return np.multiply(proba, self.weights) / sum(self.weights)

    def __str__(self):
        return "KNN"


from gmum.melm import MELM
from sklearn.datasets import load_svmlight_file
import sys
from sklearn import cross_validation
from gmum.utils import BAC
import numpy as np

if __name__ == '__main__':

    X, y = load_svmlight_file(sys.argv[1])
    X = X.toarray()

    skf = cross_validation.StratifiedKFold(y, n_folds=5)

    for clf in [
        MELM(gamma=1.0, k=2, random_state=666, n_starts=17, classifier='KNN'),
        MELM(gamma=1.0, k=2, random_state=666, n_starts=17, classifier='SVM'),
        MELM(gamma=1.0, k=2, random_state=666, n_starts=17, classifier='KDE'),
        ]:


        print clf
        scores = []
        for train_index, test_index in skf:

            clf.fit(X[train_index], y[train_index])
            score = BAC(clf.predict(X[test_index]), y[test_index])
            scores.append(score)
            print '  ', score

        print np.mean(scores), '+/-', np.std(scores)
        print

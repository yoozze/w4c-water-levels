import copy as cp

from skmultiflow.core import BaseSKMObject, RegressorMixin, MetaEstimatorMixin
from skmultiflow.trees import RegressionHoeffdingTree
from skmultiflow.utils.utils import *
from skmultiflow.utils import check_random_state


class BaggingRegression(BaseSKMObject, RegressorMixin, MetaEstimatorMixin):

    def __init__(self, base_estimator=RegressionHoeffdingTree(), n_estimators=10, random_state=None):
        super().__init__()
        # default values
        self.ensemble = None
        self.actual_n_estimators = None
        self.classes = None
        self._random_state = None  # This is the actual random_state object used internally
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.__configure()

    def __configure(self):
        if hasattr(self.base_estimator, "reset"):
            self.base_estimator.reset()
        self.actual_n_estimators = self.n_estimators
        self.ensemble = [cp.deepcopy(self.base_estimator) for _ in range(self.actual_n_estimators)]
        self._random_state = check_random_state(self.random_state)

    def reset(self):
        self.__configure()
        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):

        """if self.classes is None:
            if classes is None:
                raise ValueError("The first partial_fit call should pass all the classes.")
            else:
                self.classes = classes

        if self.classes is not None and classes is not None:
            if set(self.classes) == set(classes):
                pass
            else:
                raise ValueError("The classes passed to the partial_fit function differ from those passed earlier.")"""

        # self.__adjust_ensemble_size()
        r, _ = get_dimensions(X)
        for j in range(r):
            for i in range(self.actual_n_estimators):
                k = self._random_state.poisson()
                if k > 0:
                    for b in range(k):
                        # try:
                        self.ensemble[i].partial_fit([X[j]], [y[j]], sample_weight)
                        # except:
                        #     print('crash!')
                        #     print(list(X[j]))
                        #     print(y[j])
        return self

    def __adjust_ensemble_size(self):
        if len(self.classes) != len(self.ensemble):
            if len(self.classes) > len(self.ensemble):
                for i in range(len(self.ensemble), len(self.classes)):
                    self.ensemble.append(cp.deepcopy(self.base_estimator))
                    self.actual_n_estimators += 1

    def predict(self, X):

        return self.calculate_mean(X)

    def calculate_mean(self, X):

        r, c = get_dimensions(X)
        predict = np.zeros(r)
        for i in range(self.actual_n_estimators):
            partial_predict = self.ensemble[i].predict(X)
            predict = predict + partial_predict
        predict = predict / self.actual_n_estimators
        return predict

    def predict_proba(self, X):
        raise NotImplementedError

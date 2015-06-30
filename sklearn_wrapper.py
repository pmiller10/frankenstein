from abstract_model import AbstractModel
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
import logging
from sklearn.metrics import accuracy_score

class SkLearnWrapper(AbstractModel):


    def fit(self, data, targets, hyper_params):
        self.model.set_params(**hyper_params)
        self.model.fit(data, targets)


    def _predict(self, data):
        return self.model.predict(data)


    def score(self, preds, targets):
        return accuracy_score(targets, preds)


    def create_datasets(self, data, targets):
        train_data, cv_data = data[::2], data[1::2]
        train_targets, cv_targets = targets[::2], targets[1::2]
        return train_data, cv_data, train_targets, cv_targets



class LinearRegressionModel(SkLearnWrapper):

    """
    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    """


    def __init__(self, score_type, log_level=logging.DEBUG):
        self.score_type = score_type
        super(self.__class__, self).__init__(log_level)


    def _possible_hyper_params(self):
        return [{'normalize': True}, {'normalize': False}]



class LogisticRegressionModel(SkLearnWrapper):

    """
    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
        intercept_scaling=1, max_iter=100, multi_class='ovr',
        penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
        verbose=0)
    """

    def __init__(self, score_type, log_level=logging.DEBUG):
        self.score_type = score_type
        super(self.__class__, self).__init__(log_level)


    def _initialize_model(self):
        self.model = LogisticRegression()


    def _possible_hyper_params(self):
        params = []
        for penalty in ['l1', 'l2']:
            for i in range(5):
                c = (i * 2) + 1 
                for j in [1., 0.2, 0.1, 0.01, 0.001, 0.0001, 0.00001]:
                    tol = j
                    params.append({'C': c, 'tol': tol, 'penalty': penalty})
        return params



# TODO maybe use the .get_params method to find a way to dynamically write the _possible_hyper_params method
# then you can do the same thing with the name and save a lot of lines
class SVCModel(SkLearnWrapper):

    """
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
        kernel='rbf', max_iter=-1, probability=False, random_state=None,
        shrinking=True, tol=0.001, verbose=False)
    """


    def __init__(self, score_type, log_level=logging.DEBUG):
        self.score_type = score_type
        super(self.__class__, self).__init__(log_level)


    def _initialize_model(self):
        self.model = SVC()


    def _possible_hyper_params(self):
        params = []
        costs = [(i + 1.) for i in range(10)]
        degree = [i for i in range(5)]
        tolerance = [.1 ** i for i in range(5)]
        for c in costs:
            for d in degree:
                for tol in tolerance:
                    params.append({'C': c, 'tol': tol, 'degree': d})
        return params

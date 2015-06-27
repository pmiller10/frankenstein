from abstract_model import AbstractModel
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
import logging

class SkLearnWrapper(AbstractModel):


    def fit(self, data, targets, hyper_params):
        # TODO everytime this gets runin .fit, the model needs to be a new instance
        # otherwise, you won't be training it from scratch
        self.model.set_params(**hyper_params)
        self.model.fit(data, targets)


    def _predict(self, data):
        return self.model.predict(data)


    def score(self, preds, targets):
        errors = [(((p - t) ** 2) ** .5) for p,t in zip(preds, targets)]
        avg_error = sum(errors)/float(len(errors))
        return avg_error


    def create_datasets(self, data, targets):
        train_data, cv_data = data[::2], data[1::2]
        train_targets, cv_targets = targets[::2], targets[1::2]
        return train_data, cv_data, train_targets, cv_targets



class LinearRegressionModel(SkLearnWrapper):

    """
    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    """


    def __init__(self, log_level=logging.DEBUG):
        self.model = LinearRegression()  # TODO everytime this gets runin .fit, the model needs to be a new instance
                                         # otherwise, you won't be training it from scratch
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


    def __init__(self, log_level=logging.DEBUG):
        self.model = LogisticRegression()  # TODO everytime this gets runin .fit, the model needs to be a new instance
                                         # otherwise, you won't be training it from scratch
        super(self.__class__, self).__init__(log_level)


    def _possible_hyper_params(self):
        return [{'penalty': 'l1'}, {'penalty': 'l2'}]



# TODO maybe use the .get_params method to find a way to dynamically write the _possible_hyper_params method
# then you can do the same thing with the name and save a lot of lines
class SVCModel(SkLearnWrapper):

    """
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
        kernel='rbf', max_iter=-1, probability=False, random_state=None,
        shrinking=True, tol=0.001, verbose=False)
    """


    def __init__(self, log_level=logging.DEBUG):
        self.model = SVC()  # TODO everytime this gets runin .fit, the model needs to be a new instance
                                         # otherwise, you won't be training it from scratch
        super(self.__class__, self).__init__(log_level)


    def _possible_hyper_params(self):
        return [{'shrinking': True}, {'shrinking': False}]

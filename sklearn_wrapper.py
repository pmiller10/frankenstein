from abstract_model import AbstractModel
from sklearn.linear_model import LinearRegression, LogisticRegression
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


    def __init__(self, log_level=logging.DEBUG):
        self.model = LinearRegression()  # TODO everytime this gets runin .fit, the model needs to be a new instance
                                         # otherwise, you won't be training it from scratch
        super(self.__class__, self).__init__(log_level)


    def _possible_hyper_params(self):
        return [{'normalize': True}, {'normalize': False}]



class LogisticRegressionModel(SkLearnWrapper):


    def __init__(self, log_level=logging.DEBUG):
        self.model = LogisticRegression()  # TODO everytime this gets runin .fit, the model needs to be a new instance
                                         # otherwise, you won't be training it from scratch
        super(self.__class__, self).__init__(log_level)


    def _possible_hyper_params(self):
        return [{'penalty': 'l1'}, {'penalty': 'l2'}]

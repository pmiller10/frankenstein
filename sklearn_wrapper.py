from abstract_model import AbstractModel
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
import logging

class SkLearnWrapper(AbstractModel):


    def __init__(self, klass, default_hyperparams={}, log_level=logging.DEBUG):
        self.klass = klass
        self.name = self.__class__.__name__ + '.' + self.klass.__name__
        super(self.__class__, self).__init__(default_hyperparams, log_level)


    def fit(self, data, targets, hyperparams):
        if not len([self.model]):
            self.logger.error('No model assigned')
            raise Exception('No model')
        if hyperparams:
            self.model.set_params(**hyperparams)
        self.model.fit(data, targets)


    def _predict(self, data):
        preds = self.model.predict(data)
        return preds[0]


    def predict_proba(self, data):
        return [self.model.predict_proba(d) for d in data]


    def create_datasets(self, data, targets):
        train_data, cv_data = data[::2], data[1::2]
        train_targets, cv_targets = targets[::2], targets[1::2]
        return train_data, cv_data, train_targets, cv_targets


    def _initialize_model(self, hyperparams):
        self.model = self.klass(**hyperparams)

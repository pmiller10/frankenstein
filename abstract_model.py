import logging
from constants import Objective
from collections import defaultdict
import numpy

class AbstractModel(object):

    def __init__(self, log_level=logging.DEBUG):
        self.model = None
        self.hyper_params = None
        self.hyper_params_scores = list()

        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(log_level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        fh = logging.FileHandler('log.txt')
        fh.setLevel(log_level)
        fh.setFormatter(formatter)

        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
        self.logger = logger


    def fit(self, data, targets, hyper_params):
        raise NotImplementedError


    def predict(self, data):
        """
        param data: a list of lists
        each sublist should represent a single observation
        """
        return [self._predict(d) for d in data]


    def _predict(self, data):
        """
        param data: a list representing a single observation
        """
        raise NotImplementedError


    def predict_proba(self, data):
        raise NotImplementedError


    def optimize(self, data, targets):
        train_data, cv_data, train_targets, cv_targets = self.create_datasets(data, targets)
        for params in self._possible_hyper_params():
            score = self.cross_validate(train_data, cv_data, train_targets, cv_targets, params)
            self.logger.debug("{0}: {1}".format(params, score))
            self.hyper_params_scores.append((params, score))
        self.hyper_params, self.best_score = self._best_hyper_params()
        self.logger.info("best params {0}: {1}".format(self.hyper_params, self.best_score))


    def create_datasets(self, data, targets):
        raise NotImplementedError


    def cross_validate(self, train_data, cv_data, train_targets, cv_targets, hyper_params):
        self._initialize_model()
        self.fit(train_data, train_targets, hyper_params)
        preds = self.predict(cv_data)
        return self.score(preds, cv_targets)


    def _initialize_model(self):
        """
        This should initialize the weights of the model in place.
        For example,
        self.model = sklearn.linear_model.LogisticRegression()
        """
        raise NotImplementedError


    def score(self, preds, targets):
        raise NotImplementedError


    def _possible_hyper_params(self):
        raise NotImplementedError


    def _best_hyper_params(self):
        scores = [score for params,score in self.hyper_params_scores]
        if self.objective == Objective.MINIMIZE:
            best = min(scores)
        elif self.objective == Objective.MAXIMIZE:
            best = max(scores)
        else:
            raise Exception('{0}.objective not defined'.format(self.__class__.__name__))
        for params,score in self.hyper_params_scores:
            if score == best:
                return params, score
        raise Exception('best score not found')



class AbstractEnsemble(AbstractModel):


    def optimize(self, data, targets):
        for model in self.models:
            model.optimize(data, targets)


    def predict(self, data):
        preds = [m.predict(data) for m in self.models]
        number_of_models = range(len(self.models))
        number_of_preds = range(len(data))
        weighted_preds = []
        for i in number_of_preds:
            preds_for_one = [preds[j][i] for j in number_of_models]
            weighted_pred = self._vote(preds_for_one)
            weighted_preds.append(weighted_pred)
        return weighted_preds


    def _vote(self, preds):
        return NotImplementedError


    def fit(self, data, targets, _):
        for model in self.models:
            model.fit(data, targets, model.hyper_params)  # TODO is this the best way to store state?



class ClassifierEnsemble(AbstractEnsemble):


    def __init__(self, models, voter, objective, log_level=logging.DEBUG):
        """
        param :models is a list of FML models.
        param :objective is a constants.Objective constant
        param :voter is a instance of a model to vote for the output
        """
        self.objective = objective
        self.models = []
        self.voter = voter
        self.models = models
        super(self.__class__, self).__init__(log_level)


    def fit(self, data, targets, _):
        """
        param :data should be a list of lists
        """
        for model in self.models:
            model.fit(data, targets, model.hyper_params)  # TODO split into CV set?

        preds = [m.predict_proba(data) for m in self.models]
        meta_features = self._meta_features(data)
        self.voter.optimize(meta_features, targets)
        self.voter.fit(meta_features, targets, {})


    def predict(self, data):
        meta_features = self._meta_features(data)
        return self.voter.predict(meta_features)


    def predict_proba(self, data):
        meta_features = self._meta_features(data)
        return self.voter.predict_proba(meta_features)


    def _meta_features(self, data):
        preds = [m.predict_proba(data) for m in self.models]
        meta_features = []
        number_of_models = range(len(self.models))
        number_of_preds = range(len(data))
        for j in number_of_preds:
            d = data[j]
            preds_for_one = [preds[i][j] for i in number_of_models]
            preds_for_one = numpy.concatenate(preds_for_one, axis=0)
            preds_for_one = preds_for_one.flatten()
            d = d.flatten()
            preds_for_one = numpy.concatenate((preds_for_one, d), axis=0)
            meta_features.append(preds_for_one)
        return meta_features

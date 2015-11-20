import logging
from collections import defaultdict
import numpy
from constants import Objective
from _globals import Config
import hyper_params_generator



class AbstractModel(object):

    def __init__(self, default_hyper_params={}, log_level=logging.DEBUG):
        self.model = None
        self.hyper_params = None
        self.default_hyper_params = default_hyper_params
        self.hyper_params_scores = list()

        name = self.__class__.__name__ + ":" + str(id(self))
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        msg = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(msg)

        fh = logging.FileHandler('log.txt')
        fh.set_name(self.__class__.__name__)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)

        ch = logging.StreamHandler()
        ch.set_name(self.__class__.__name__)
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
        # the model was created with defined hyperparams so don't
        # cross validate to optimize them
        if self.default_hyper_params:
            # TODO since you're going to run .fit() later,
            # then should you just not train at all and just return here?
            self.logger.info('Using default hyperparams. Skipping training.')
            score = self.cross_validate(train_data, cv_data, train_targets, cv_targets, self.default_hyper_params)
            self.hyper_params_scores.append((self.default_hyper_params, score))
        # optimize hyperparams
        else:
            for i in range(Config.epochs):
                params = hyper_params_generator.generate(self.klass)
                score = self.cross_validate(train_data, cv_data, train_targets, cv_targets, params)
                self.logger.debug("Epoch {0} score = {1}. Hyperparams: {2}".format(i, score, params))
                self.hyper_params_scores.append((params, score))
        self.hyper_params, self.best_score = self._best_hyper_params()
        self.logger.info("Best score = {0}. Hyperparams: {1}\n".format(self.best_score, self.hyper_params))


    def create_datasets(self, data, targets):
        raise NotImplementedError


    def cross_validate(self, train_data, cv_data, train_targets, cv_targets, hyper_params):
        self._initialize_model(hyper_params)
        self.fit(train_data, train_targets, hyper_params)
        preds = self.predict(cv_data)
        return self._score(preds, cv_targets)


    def _initialize_model(self, hyper_params):
        """
        This should initialize the weights of the model in place.
        For example,
        self.model = sklearn.linear_model.LogisticRegression(**hyper_params)
        """
        raise NotImplementedError


    def _score(self, preds, targets):
        return Config.loss(preds, targets)


    def _best_hyper_params(self):
        scores = [score for params,score in self.hyper_params_scores]
        objective = Config.objective
        if objective == Objective.MINIMIZE:
            best = min(scores)
        elif objective == Objective.MAXIMIZE:
            best = max(scores)
        else:
            raise Exception('{0}.objective not defined'.format(self.__class__.__name__))
        for params,score in self.hyper_params_scores:
            if score == best:
                return params, score
        raise Exception('best score not found')



class AbstractEnsemble(AbstractModel):


    def predict(self, data):
        meta_features = self._meta_features(data)
        return self.voter.predict(meta_features)


    def optimize(self, data, targets):
        for model in self.models:
            model.optimize(data, targets)


    # TODO remove the _ param
    def fit(self, data, targets, _):
        """
        param :data should be a list of lists
        """
        for model in self.models:
            model.fit(data, targets, model.hyper_params)  # TODO split into CV set?

        if self.__class__.__name__ == 'RegressionEnsemble':
            preds = [m.predict(data) for m in self.models]
        elif self.__class__.__name__ == 'ClassifierEnsemble':
            preds = [m.predict_proba(data) for m in self.models]
        else:
            msg = 'Unknown Ensemble class: {0}'.format(self.__class__)
            raise Exception(msg)

        meta_features = self._meta_features(data)
        self.logger.info('Optimizing voter')
        self.voter.optimize(meta_features, targets)
        self.voter.fit(meta_features, targets, {})


class RegressionEnsemble(AbstractEnsemble):

    def __init__(self, models, voter, log_level=logging.DEBUG):
        """
        param :models is a list of FML models.
        param :voter is a instance of a model to vote for the output
        """
        self.models = []
        self.voter = voter
        self.models = models
        super(self.__class__, self).__init__({}, log_level)


    def _meta_features(self, data):
        """
        Iterates over the models in the ensemble and returns a list
        of predictions. These are used as input features to the voter.
        """
        preds = [m.predict(data) for m in self.models]
        meta_features = []
        number_of_models = range(len(self.models))
        number_of_preds = range(len(data))
        # iterate over predictions to create list of lists,
        # where each sublist is for one particular input
        for j in number_of_preds:
            d = data[j]
            preds_for_one = [preds[i][j] for i in number_of_models]
            meta_features.append(preds_for_one)
        # TODO this returns list of lists, but
        # ClassifierEnsemble._meta_features returns list of numpy arrays
        return meta_features


class ClassifierEnsemble(AbstractEnsemble):


    def __init__(self, models, voter, log_level=logging.DEBUG):
        """
        param :models is a list of FML models.
        param :voter is a instance of a model to vote for the output
        """
        self.models = []
        self.voter = voter
        self.models = models
        super(self.__class__, self).__init__({}, log_level)


    def predict_proba(self, data):
        meta_features = self._meta_features(data)
        return self.voter.predict_proba(meta_features)


    def _meta_features(self, data):
        """
        Iterates over the models in the ensemble and returns a list
        of predictions. These are used as input features to the voter.
        """
        preds = [m.predict_proba(data) for m in self.models]
        meta_features = []
        number_of_models = range(len(self.models))
        number_of_preds = range(len(data))
        # iterate over predictions to create list of arrays,
        # where each array is for one particular input
        for j in number_of_preds:
            d = data[j]
            preds_for_one = [preds[i][j] for i in number_of_models]
            preds_for_one = numpy.concatenate(preds_for_one, axis=0)
            preds_for_one = preds_for_one.flatten()
            if hasattr(d, 'flatten'):  # check if data is numpy or list
                d = d.flatten()
            preds_for_one = numpy.concatenate((preds_for_one, d), axis=0)
            meta_features.append(preds_for_one)
        return meta_features

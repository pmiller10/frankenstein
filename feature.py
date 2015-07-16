import logging
from sklearn_wrapper import LinearRegressionModel
from constants import Objective, FeatureConstants
import lib

class Feature(object):

    def __init__(self, objective, log_level=logging.DEBUG):
        self.objective = objective
        self.polynomial = None
        self.tensor = None
        self.k_means = None

        # TODO this is adding multiple loggers to the handler
        # check this with self.logger.handlers
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


    def optimize(self, data, targets):
        model = LinearRegressionModel(Objective.MINIMIZE, log_level=self.logger.level)
        model.optimize(data, targets)
        best = model.best_score
        self.polynomial = 1
        for p in range(1, FeatureConstants.MAX_POLYNOMIAL + 1):
            new_data = lib.polynomial(data, p)
            model = LinearRegressionModel(Objective.MINIMIZE, log_level=self.logger.level)
            model.optimize(new_data, targets)
            self.logger.info("score={0} with polynomial {1}".format(model.best_score, p))
            if self.objective == Objective.MINIMIZE:
                if model.best_score < best:
                    self.polynomial = p
                    best = model.best_score
            elif self.objective == Objective.MAXIMIZE:
                if model.best_score > best:
                    self.polynomial = p
                    best = model.best_score


    def create_datasets(self, data, targets):  # TODO this is copy/pasted from sklearn_wrapper
        train_data, cv_data = data[::2], data[1::2]
        train_targets, cv_targets = targets[::2], targets[1::2]
        return train_data, cv_data, train_targets, cv_targets


    def _best_polynomial(self, data, targets):
        pass

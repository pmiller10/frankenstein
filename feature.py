import logging
from sklearn_wrapper import LinearRegressionModel, LogisticRegressionModel
from constants import Objective, FeatureConstants
import lib

class Feature(object):

    def __init__(self, score_type, log_level=logging.DEBUG):
        self.score_type = score_type
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
        # TODO 
        #for f in functions:
        #    for a in args:
        #        f.function
        #self._optimize_polynomial(data, targets)
        self._optimize_k_means(data, targets)

    def _optimize_polynomial(self, data, targets):
        model = LinearRegressionModel(self.score_type, log_level=self.logger.level)
        model.optimize(data, targets)
        best = model.best_score
        self.polynomial = FeatureConstants.MIN_POLYNOMIAL
        # TODO refactor this section into a function that takes another function as argument
        for p in range(1, FeatureConstants.MAX_POLYNOMIAL + 1):
            new_data = lib.polynomial(data, p)
            model = LinearRegressionModel(self.score_type, log_level=self.logger.level)
            model.optimize(new_data, targets)
            self.logger.info("score={0} with polynomial {1}".format(model.best_score, p))

            # TODO refactor this section into a separate function
            if self.score_type == Objective.MINIMIZE:
                if model.best_score < best:
                    self.polynomial = p
                    best = model.best_score
            elif self.score_type == Objective.MAXIMIZE:
                if model.best_score > best:
                    self.polynomial = p
                    best = model.best_score


    def _optimize_k_means(self, data, targets):
        model = LinearRegressionModel(self.score_type, log_level=self.logger.level)
        model.optimize(data, targets)
        best = model.best_score
        self.k_means = FeatureConstants.MIN_K_MEANS
        # TODO refactor this section into a function that takes another function as argument
        for k in range(FeatureConstants.MIN_K_MEANS, FeatureConstants.MAX_K_MEANS + 1):
            new_data = lib.k_means(data, k)
            #import pdb; pdb.set_trace()
            print new_data[0]
            print new_data[1]
            print new_data[2]
            model = LinearRegressionModel(self.score_type, log_level=self.logger.level)
            model.optimize(new_data, targets)
            self.logger.info("score={0} with k means {1}".format(model.best_score, k))

            # TODO refactor this section into a separate function
            if self.score_type == Objective.MINIMIZE:
                if model.best_score < best:
                    print '\n\n'
                    print 'old', best, 'new', model.best_score
                    print 'setting k means', k
                    self.k_means = k
                    best = model.best_score
            elif self.score_type == Objective.MAXIMIZE:
                if model.best_score > best:
                    print '\n\n'
                    print 'old', best, 'new', model.best_score
                    print 'setting k means', k
                    self.k_means = k
                    best = model.best_score



    def create_datasets(self, data, targets):  # TODO this is copy/pasted from sklearn_wrapper
        train_data, cv_data = data[::2], data[1::2]
        train_targets, cv_targets = targets[::2], targets[1::2]
        return train_data, cv_data, train_targets, cv_targets


    def _best_polynomial(self, data, targets):
        pass



class PreprocessingFunction(object):  # TODO used a namedtuple


    def __init__(self, name, function, args):
        self.name, self.function, self.args = name, function, args

from sklearn_wrapper import LinearRegressionModel
from constants import ScoreType
import lib

class Feature(object):

    def __init__(self, score_type):
        self.score_type = score_type
        self.polynomial = None
        self.tensor = None
        self.k_means = None


    def optimize(self, data, targets):
        model = LinearRegressionModel(ScoreType.MINIMIZE)
        model.optimize(data, targets)
        best = model.best_score
        self.polynomial = 1
        for p in range(1, 3):
            new_data = lib.polynomial(data, p)
            model = LinearRegressionModel(ScoreType.MINIMIZE)
            model.optimize(new_data, targets)
            new_best = model.best_score
            if self.score_type == ScoreType.MINIMIZE:
                if model.best_score < best:
                    self.polynomial = p
            elif self.score_type == ScoreType.MAXIMIZE:
                if model.best_score > best:
                    self.polynomial = p


    def create_datasets(self, data, targets):  # TODO this is copy/pasted from sklearn_wrapper
        train_data, cv_data = data[::2], data[1::2]
        train_targets, cv_targets = targets[::2], targets[1::2]
        return train_data, cv_data, train_targets, cv_targets


    def _best_polynomial(self, data, targets):
        pass

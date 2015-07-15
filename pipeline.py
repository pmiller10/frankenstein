import logging
from constants import Objective



class Pipeline(object):


    def __init__(self, transformer, model, objective, log_level=logging.DEBUG): 
        self.transformer = transformer
        self.model = model
        self.objective = objective
        self.log_level = log_level



    def fit(self, data, targets, extra_data=None):
        """
        Should accept a dataset and targets.
        Should set self.hyper_params
        """

        # test without transforming data to get a baseline score
        model = self.model(self.objective, log_level=self.log_level)
        model.optimize(data, targets)
        best = model.best_score
        self.hyper_params = None
        print model.best_score, 1

        # the transformer class should return a generator of modified datasets
        # test each one with the model, and if the performance is better, then
        # set the new best score to that performance
        for transformed, hyper_params in self.transformer().each_transformation(data, extra_data):
            model = self.model(self.objective, log_level=self.log_level)
            model.optimize(transformed, targets)
            print model.best_score, hyper_params

            if self.objective == Objective.MINIMIZE:
                if model.best_score < best:
                    best = model.best_score
                    self.hyper_params = hyper_params
            elif self.objective == Objective.MAXIMIZE:
                if model.best_score > best:
                    best = model.best_score
                    self.hyper_params = hyper_params



    def transform(self, data):
        return self.transformer().transform(data, **self.hyper_params)

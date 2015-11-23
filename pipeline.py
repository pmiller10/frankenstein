import logging
from constants import Objective
from _globals import Config



class Pipeline(object):


    def __init__(self, transformer, model, log_level=logging.DEBUG):
        """
        The transformer should be a Preprocess class.
        The model should be a FML compliant model.
        """

        self.transformer = transformer
        self.model = model
        self.log_level = log_level

        # set up logger
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



    def fit(self, data, targets, unsupervised_data=None):
        """
        Should accept a dataset and targets.
        Should set self.hyper_params
        """

        # test without transforming data to get a baseline score
        self.model.optimize(data, targets)
        best = self.model.best_score
        self.hyper_params = None
        self.logger.info('Score without transformation = {0}'.format(self.model.best_score))

        # the transformer class should return a generator of modified datasets
        # test each one with the self.model, and if the performance is better, then
        # set the new best score to that performance
        for transformed, hyper_params in self.transformer().each_transformation(data, unsupervised_data):

            # Pass the params from the model to the class of the model.
            # This creates a new model with fresh weights that haven't been
            # fitted yet. Don't pop 'default_hyperparams' so that the model
            # can be created with default params.
            model_params = self.model.__dict__
            model_params.pop('best_score')
            model_params.pop('hyper_params')
            model_params.pop('hyper_params_scores')
            model_params.pop('logger')
            model_params.pop('model')
            self.model = self.model.__class__(**model_params)
            self.model.optimize(transformed, targets)
            self.logger.info("Best score with {0} = {1}".format(hyper_params, self.model.best_score))

            objective = Config.objective
            if objective == Objective.MINIMIZE:
                if self.model.best_score < best:
                    best = self.model.best_score
                    self.hyper_params = hyper_params
            elif objective == Objective.MAXIMIZE:
                if self.model.best_score > best:
                    best = self.model.best_score
                    self.hyper_params = hyper_params



    def transform(self, data):
        if self.hyper_params:
            return self.transformer().transform(data, **self.hyper_params)
        else:
            self.logger.warn('hyper_params set to None. Either you forgot to run .fit(), or .fit() found no improvement')
            return data

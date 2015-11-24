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
        self.name = self.__class__.__name__ + '.' + self.transformer.__name__

        # set up logger
        name = self._model_name()
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        extra = {'model_name': self._model_name()}

        msg = '[%(asctime)s] %(levelname)s [%(model_name)s.%(funcName)s:%(lineno)d] %(message)s'
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

        logger = logging.LoggerAdapter(logger, extra)
        self.logger = logger


    def _model_name(self):
        # For logging: allows the model name to be safely overridden.
        if hasattr(self, 'name'):
            if self.name:
                return self.name + '(' + str(id(self)) + ')'
        else:
            return self.__class__.__name__ + '(' + str(id(self)) + ')'


    def fit(self, data, targets):
        """
        Should accept a dataset and targets.
        Should set self.hyperparams
        """

        # test without transforming data to get a baseline score
        self.model.tune(data, targets)
        best = self.model.best_score
        self.hyperparams = None
        self.logger.info('Score without transformation = {0}'.format(self.model.best_score))

        # the transformer class should return a generator of modified datasets
        # test each one with the self.model, and if the performance is better, then
        # set the new best score to that performance
        for transformed, hyperparams in self.transformer().each_transformation(data):

            # Pass the params from the model to the class of the model.
            # This creates a new model with fresh weights that haven't been
            # fitted yet. Don't pop 'default_hyperparams' so that the model
            # can be created with default params.
            model_params = self.model.__dict__
            blacklist = ['best_score', 'hyperparams', 'hyperparams_scores',
                         'logger', 'model', 'name']
            for param in blacklist:
                model_params.pop(param)

            try:
                self.model = self.model.__class__(**model_params)
            except TypeError as ee:
                msg = """Tried to create model but there's probably a param
                         that needs to be included in the blacklist above.
                         Actual error: {0}""".format(ee)
                raise Exception(msg)


            self.model.tune(transformed, targets)
            self.logger.info("Best score with {0} = {1}".format(hyperparams, self.model.best_score))

            objective = Config.objective
            if objective == Objective.MINIMIZE:
                if self.model.best_score < best:
                    best = self.model.best_score
                    self.hyperparams = hyperparams
            elif objective == Objective.MAXIMIZE:
                if self.model.best_score > best:
                    best = self.model.best_score
                    self.hyperparams = hyperparams
        self.logger.info("Best overall score {0}".format(self.hyperparams))



    def transform(self, data, force_hyperparams={}):
        # If user explicity defines defines the params, then use them.
        if force_hyperparams:
            msg = 'Using forced params: {0}'.format(force_hyperparams)
            self.logger.info(msg)
            return self.transformer().transform(data, **force_hyperparams)

        # Otherwise, use the trained params.
        if self.hyperparams:
            msg = 'Using trained params: {0}'.format(self.hyperparams)
            return self.transformer().transform(data, **self.hyperparams)
        else:
            self.logger.warning('hyperparams set to None. Either you forgot to run .fit(), or .fit() found no improvement')
            return data

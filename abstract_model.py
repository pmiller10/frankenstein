import logging

class AbstractModel():

    def __init__(self, log_level=logging.DEBUG):
        self.hyper_params = None
        self.hyper_params_scores = list()

        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(log_level)
        fh = logging.FileHandler('log.txt')
        fh.setLevel(log_level)
        logger.addHandler(fh)
        self.logger = logger

    def fit(self, data, targets):
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

    def optimize(self, data, targets):
        train_data, cv_data, train_targets, cv_targets = self.create_datasets(data, targets)
        for params in self._possible_hyper_params():
            score = self.cross_validate(train_data, cv_data, train_targets, cv_targets)
            logger.info(params, score)
            self.hyper_params.append((params, score))
        self.hyper_params = self._best_hyper_params()

    def create_datasets(self, data, targets):
        self.logger.debug('hello')
        raise NotImplementedError

    def cross_validate(self, train_data, cv_data, train_targets, cv_targets):
        self.fit(train_data, cv_data)
        preds = self.predict(cv_targets)
        return self.score(preds, cv_targets)

    def _possible_hyper_params(self):
        raise NotImplementedError

    def _best_hyper_params(self):
        scores = [score for params,score in self.hyper_params_scores]
        best = min(scores)
        for params,score in self.hyper_params_scores:
            if score == best:
                self.hyper_params = params
                return


if __name__ == "__main__":
    model = AbstractModel()
    model.logger.debug('hello')
    #model.optimize([], [])

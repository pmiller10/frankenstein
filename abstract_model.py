import logging
from constants import ScoreType
from collections import defaultdict

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


    def optimize(self, data, targets):
        train_data, cv_data, train_targets, cv_targets = self.create_datasets(data, targets)
        for params in self._possible_hyper_params():
            score = self.cross_validate(train_data, cv_data, train_targets, cv_targets, params)
            self.logger.info("{0}: {1}".format(params, score))
            self.hyper_params_scores.append((params, score))
        self.hyper_params = self._best_hyper_params()


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
        if self.score_type == ScoreType.MINIMIZE:
            best = min(scores)
        elif self.score_type == ScoreType.MAXIMIZE:
            best = max(scores)
        else:
            raise Exception('{0}.score_type not defined'.format(self.__class__.__name__))
        for params,score in self.hyper_params_scores:
            if score == best:
                return params
        raise Exception('best score not found')
 


class AbstractEnsemble(AbstractModel):


    #def __init__(self, models, score_type, log_level=logging.DEBUG):
        #self.models = models
        #self.score_type = score_type
        #super(self.__class__, self).__init__(log_level)

    
    def optimize(self, data, targets):
        for m in self.models:
            print len(data)
            m.optimize(data, targets)

    
    #def _predict(self, data):
        #preds = [m.predict(data) for m in self.models]
        #return self._vote(preds) 


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
        for m in self.models:
            print len(data)
            m.fit(data, targets, m.hyper_params)



class ClassifierEnsemble(AbstractEnsemble):


    def __init__(self, models, score_type, log_level=logging.DEBUG):
        self.models = models  # TODO should you init the model classes here, rather than be passed the isntances?
        self.score_type = score_type
        super(self.__class__, self).__init__(log_level)


    def _vote(self, preds):
        votes = defaultdict(int)
        for p in preds:
            votes[p] += 1
        most_votes = max(votes.values())
        for klass,vote_count in votes.items():
            if vote_count == most_votes:
                return klass

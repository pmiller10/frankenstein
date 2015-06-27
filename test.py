from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import datasets
from abstract_model import AbstractModel
import logging



class LinearModel(AbstractModel):

    def fit(self, data, targets, hyper_params):
        model = LogisticRegression()
        model.set_params(**hyper_params)
        model.fit(data, targets)
        self.model = model
        self.logger.debug(self.model)


    def _predict(self, data):
        return self.model.predict(data)


    def score(self, preds, targets):
        errors = [(((p - t) ** 2) ** .5) for p,t in zip(preds, targets)]
        avg_error = sum(errors)/float(len(errors))
        return avg_error


    def create_datasets(self, data, targets):
        train_data, cv_data = data[::2], data[1::2]
        train_targets, cv_targets = targets[::2], targets[1::2]
        return train_data, cv_data, train_targets, cv_targets


    def _possible_hyper_params(self):
        #return [{'normalize': True}, {'normalize': False}]
        return [{'penalty': 'l1'}, {'penalty': 'l2'}]



class TestModel():

    def test_score(self):
        model = LinearModel(log_level=logging.WARN)
        preds, targets = range(10), range(10)
        score = model.score(preds, targets)
        assert(score == 0)

        preds, targets = [0], [1]
        score = model.score(preds, targets)
        assert(score == 1)

        preds, targets = range(10), range(10, 20)
        score = model.score(preds, targets)
        assert(score == 10)


    def test_datasets(self):
        model = LinearModel(log_level=logging.WARN)
        data = [[i] for i in range(10)]
        targets = range(10)

        train_data, cv_data, train_targets, cv_targets = model.create_datasets(data, targets)

        expected_data = []
        for a,b in zip(train_data, cv_data):
            expected_data.append(a)
            expected_data.append(b)

        expected_targets = []
        for a,b in zip(train_targets, cv_targets):
            expected_targets.append(a)
            expected_targets.append(b)

        assert(data == expected_data) 
        assert(targets == expected_targets) 


    def test_optimize(self):
        model = LinearModel(log_level=logging.WARN)
        data = [[i] for i in range(10)]
        targets = range(10, 20)
        model.optimize(data, targets)


    def test_optimize(self):
        model = LinearModel(log_level=logging.WARN)
        assert(model.hyper_params == None)
        assert(model.hyper_params_scores == [])

        iris = datasets.load_iris()
        data = list(iris.data)
        targets = list(iris.target)
        model.optimize(data, targets)

        #import pdb; pdb.set_trace()
        assert(model.hyper_params == {'penalty': 'l1'})
        assert(len(model.hyper_params_scores) == 2)



 
if __name__ == "__main__":
    TestModel().test_score()
    TestModel().test_datasets()
    TestModel().test_optimize()
    TestModel().test_optimize()
    print 'success'

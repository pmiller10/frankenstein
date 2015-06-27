from sklearn.linear_model import LinearRegression
from abstract_model import AbstractModel



class LinearModel(AbstractModel):

    def fit(self, data, targets, hyper_params):
        model = LinearRegression()
        model.set_params(**hyper_params)
        model.fit(data, targets)
        self.model = model
        self.logger.debug(self.model)

    def _predict(self, data):
        return self.model.predict(data)

    def score(self, preds, targets):
        errors = [(((p - t) ** 2) ** .5) for p,t in zip(preds, targets)]
        return sum(errors)/float(len(errors))

    def create_datasets(self, data, targets):
        train_data, cv_data = data[::2], data[1::2]
        train_targets, cv_targets = targets[::2], targets[1::2]
        return train_data, cv_data, train_targets, cv_targets

    def _possible_hyper_params(self):
        return [{'normalize': True}, {'normalize': False}]



class TestModel():

    def test_score(self):
        model = LinearModel()
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
        model = LinearModel()
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
        model = LinearModel()
        data = [[i] for i in range(10)]
        targets = range(10, 20)
        model.optimize(data, targets)


 
if __name__ == "__main__":
    TestModel().test_score()
    TestModel().test_datasets()
    TestModel().test_optimize()

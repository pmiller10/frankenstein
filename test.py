from sklearn import datasets
import logging
from sklearn_wrapper import LogisticRegressionModel



class TestModel():

    def test_score(self):
        model = LogisticRegressionModel(log_level=logging.WARN)
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
        model = LogisticRegressionModel(log_level=logging.WARN)
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
        model = LogisticRegressionModel(log_level=logging.WARN)
        assert(model.hyper_params == None)
        assert(model.hyper_params_scores == [])

        iris = datasets.load_iris()
        data = list(iris.data)
        targets = list(iris.target)
        model.optimize(data, targets)

        assert(model.hyper_params == {'penalty': 'l1'})
        assert(len(model.hyper_params_scores) == 2)



 
if __name__ == "__main__":
    TestModel().test_score()
    TestModel().test_datasets()
    TestModel().test_optimize()
    print 'success'

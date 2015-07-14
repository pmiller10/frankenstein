from sklearn import datasets
import logging
from sklearn_wrapper import LogisticRegressionModel
from constants import Objective
from feature import Feature
import lib



class TestModel():

    def test_score(self):
        model = LogisticRegressionModel(Objective.MINIMIZE, log_level=logging.WARN)
        preds, targets = range(10), range(10)
        score = model.score(preds, targets)
        assert(score == 1.)

        preds, targets = [0], [1]
        score = model.score(preds, targets)
        assert(score == 0.)

        preds, targets = range(10), range(10, 20)
        score = model.score(preds, targets)
        assert(score == 0.)


    def test_datasets(self):
        model = LogisticRegressionModel(Objective.MINIMIZE, log_level=logging.WARN)
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
        model = LogisticRegressionModel(Objective.MINIMIZE, log_level=logging.WARN)
        assert(model.hyper_params == None)
        assert(model.hyper_params_scores == [])

        iris = datasets.load_iris()
        data = list(iris.data)
        targets = list(iris.target)
        model.optimize(data, targets)

        assert(len(model.hyper_params_scores) == 70)



class TestFeature():

    def _test_polynomial(self, polynomial):
        data = range(10)
        data = [float(d) for d in data]
        targets = [d**polynomial for d in data]
        data = [[d] for d in data]

        feature = Feature(Objective.MINIMIZE)
        feature.optimize(data, targets)

        assert feature.polynomial == polynomial, "{0} != {1}".format(feature.polynomial, polynomial)

    def test_optimize_higher_order_polynomial(self):
        self._test_polynomial(2)
        self._test_polynomial(3)
        self._test_polynomial(4)
        self._test_polynomial(5)



class TestLib():

    def test_polynomial(self):
        n = 4
        poly = 2
        data, expected = range(n), range(n)
        squared = [e**poly for e in expected]
        expected.extend(squared)

        output = lib.polynomial(data, poly)

        leftover = (output-expected).any()
        assert leftover == False, "{0} != {1}".format(output, expected)


    def test_norm_positives(self):
        data = [0., 5., 10.]
        expected = [0., .5, 1.]
        output = lib.norm(data)
        leftover = (output-expected).any()
        assert leftover == False, "{0} != {1}".format(output, expected)
        

    def test_norm_negatives(self):
        data = [-10., 0., 10.]
        expected = [0., .5, 1.]
        output = lib.norm(data)
        leftover = (output-expected).any()
        assert leftover == False, "{0} != {1}".format(output, expected)

    
    def test_scaler_vector(self):
        data = [-1., 0., 1]
        output = lib.scale(data)
        assert sum(output) == 0


    def test_scaler_matrix(self):
        d1 = [-10., 0., 50.]
        d2 = [-100., 0., 100.]
        data = [d1, d2]
        output = lib.scale(data)
        assert sum(sum(output)) == 0


 
if __name__ == "__main__":
    TestModel().test_score()
    TestModel().test_datasets()
    TestModel().test_optimize()
    TestLib().test_polynomial()
    TestLib().test_norm_positives()
    TestLib().test_norm_negatives()
    TestLib().test_scaler_vector()
    TestLib().test_scaler_matrix()
    TestFeature().test_optimize_higher_order_polynomial()
    print 'success'

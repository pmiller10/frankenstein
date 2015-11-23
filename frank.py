from constants import Objective
from data import Data
from pipeline import Pipeline
from preprocess import Polynomial
from abstract_model import ClassifierEnsemble
from sklearn_wrapper import  LinearRegressionModel, LogisticRegressionModel, SVCModel
from score import score
import logging


"""
This is supposed to be a sample interface for me to develop to support.
"""

objective = Objective.MAXIMIZE

# data
train_data, train_targets = Data.train()
test_data, test_targets = Data.test()

# feature engineering
extra_data = test_data
pipe = Pipeline(Polynomial, LogisticRegressionModel, objective, logging.WARN)
pipe.fit(train_data, train_targets, extra_data)
print pipe.hyperparams

# train model
train_data = pipe.transform(train_data)

voter1 = LogisticRegressionModel(objective, logging.INFO)
models = [m(objective, logging.INFO) for m in [SVCModel, LogisticRegressionModel]]
ensemble = ClassifierEnsemble(models, voter1, objective, logging.INFO)

voter2 = LogisticRegressionModel(objective, logging.INFO)
master = ClassifierEnsemble([ensemble], voter2, objective, logging.INFO)

master.optimize(train_data, train_targets)
master.fit(train_data, train_targets, master.hyperparams)

# submission file
test_data = pipe.transform(test_data)
preds = master.predict(test_data)
print score(preds, test_targets)

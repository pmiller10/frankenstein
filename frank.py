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
pipe = Pipeline(Polynomial, SVCModel, objective, logging.WARN)
pipe.fit(train_data, train_targets, extra_data)
print pipe.hyper_params

# train model
train_data = pipe.transform(train_data)
voter = LogisticRegressionModel
model = ClassifierEnsemble([LogisticRegressionModel, SVCModel], voter, objective, logging.INFO)
model.optimize(train_data, train_targets)
model.fit(train_data, train_targets, model.hyper_params)

# submission file
test_data = pipe.transform(test_data)
preds = model.predict(test_data)
print score(preds, test_targets)

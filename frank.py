from constants import Objective
from data import Data
from pipeline import Pipeline
from preprocess import Polynomial
from abstract_model import AbstractEnsemble
from sklearn_wrapper import  LinearRegressionModel, LogisticRegressionModel, SVCModel
from submission import submission


"""
This is supposed to be a sample interface for me to develop to support.
"""

objective = Objective.MAXIMIZE

# data
train_data, train_targets = Data.train()
test_data, test_targets = Data.test()

# feature engineering
extra_data = test_data
pipe = Pipeline([Polynomial, LogisticRegressionModel], objective)
pipe.fit(train_data, train_targets, extra_data)

# train model
train_data = pipe.transform(train_data)
model = Ensemble([LogisticRegressionModel, SVCModel], objective)
model.fit(train_data, train_targets)

# submission file
preds = model.predict(test_data)
submission(preds)

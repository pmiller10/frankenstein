# Frankenstein Machine Learning (FML)

A machine learning framework for developing predictive models. This was originally intended to automate Kaggle competitions, but it can be used more generally. The framework is designed to make ensembling and cross validation very easy. Why Frankenstein? It has a very extensible interface to allow models from various Python libraries to be patched together into one ensemble very easily. This is still under development, but works well already with simple datasets.


## Overview

FML treats each model of an ensemble as a hyperparameter and then cross validates the ensemble and automatically select which models to use. Before the ensemble is cross validated, various preprocessing and feature engineering steps can be cross validated as well.

The high-level steps to create a predictive model are

1. Preprocessing and feature engineering (e.g. stemming, normalization, n-grams)
2. Model selection and tuning (e.g. cross validating to tune hyper parameters)

Step 1 is represented by Preprocess classes. A Preprocess class iterates over variations of a specific preprocessing step. Preprocess subclasses make a series of transformations to the dataset, which are passed in a generator to the pipeline object, which selects the best one to use.

For example, you could create a Polynomial class, which provides a generator that raises the dataset to a higher-order polynomial. The Pipeline would then iterate over these and train a model on each one, testing which has the best performance. Another Preprocess subclass could yield two versions of the same text dataset, one where the text is stemmed and another where it is not.

Multiple Pipelines can (and probably should) be chained together to evaluate a series of steps. The Pipeline(s) then apply the tranformation steps and the resultant data is then passed to a model.

Step 2 is represented by the Ensemble classes (ClassifierEnsemble and RegressionEnsemble). You can think of each model as a hyperparameter for the ensemble. Then the ensemble can be cross validated, tuning the voting weight for each model, and each model is cross validated.

When using the Ensemble, you are not limited to models provided by the FML framework. Any model with the correct API can be used. This is intentionally based off of scikit-learn to make it easy to use those out of the box. However, if you want to use another model that doesn't follow this API, there is information below on how to do that (TODO).


## Example

```
from FML.pipeline import Pipeline
from FML.preprocess import Polynomial
from FML.abstract_model import ClassifierEnsemble, RegressionEnsemble
from FML.sklearn_wrapper import SkLearnWrapper
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import datasets


# Use scikit-learn dataset
iris = datasets.load_iris()
data = list(iris.data)
targets = list(iris.target)
train_data, holdout_data = data[::2], data[1::2]
train_targets, holdout_targets = targets[::2], targets[1::2]

# Feature engineering: Raise the training data to a
# higher-order polynomial to see which polynomial does best
model = SkLearnWrapper(LogisticRegression)
pipe = Pipeline(Polynomial, model)
pipe.fit(train_data, train_targets)
train_data = pipe.transform(train_data)

# Use scikit-learn models with wrapper class
models = [LogisticRegression, RandomForestClassifier, SVC]
models = [SkLearnWrapper(m) for m in models]
voter = SkLearnWrapper(LogisticRegression)
ensemble = RegressionEnsemble(models, voter)

# Cross validate each model in the ensemble
ensemble.optimize(train_data, train_targets)
# Fit each model with hyperparams from the cross validation
ensemble.fit(train_data, train_targets)

# Predict on unseen data
holdout_data = pipe.transform(holdout_data)
preds = ensemble.predict(holdout_data)
```


## Configuration

FML expects there to be config.py file in whatever directory you're exectuting your app/script from. The loss function you optimize can be configured here, as well as any Preprocess steps you want to cross validate. The file should look something like this:

```
from FML.default_config import DefaultConfig

class Config(DefaultConfig):

    @classmethod
    def loss(cls, preds, targets):
        return sum([(p-t)**2 for p,t in zip(preds, targets)])

    epochs = 10  # number of hyperparam combinations to cross validate per model
```

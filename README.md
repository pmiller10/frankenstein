# Frankenstein Machine Learning (FML)

This is a machine learning framework for developing predictive models.
This was originally intended to automate Kaggle competitions, but it can be used more generally.

# Setup

Currently dependent on sklearn.

# Overview

The idea is that much of the process of developing a predictive model can be automated if each step is cross validated. First, cross validate a series of preprocessing/feature-engineering steps. Then apply the best preprocessing steps to the dataset. Then cross validate an ensemble of multiple models, where the ensemble selects which models to use or discard.

The high-level steps to create any predictive model are

1. Preprocessing (e.g. stemming, lemmatizing, normalization)
2. Feature engineering (e.g. n-grams, higher-order polynomials)
3. Model selection and tuning (e.g. cross validating to tune hyper parameters)

The line betweeen preprocessing and feature engineering is always a bit blurry, and FML handles them both the same way. Steps 1 and 2 are represented by Preprocess classes. A Preprocess class iterates over variations of a specific preprocessing step.

Subclasses of the Preprocess class are passed to an instance of the Pipeline class. A Pipeline object then iterates over the various transformations provided by the Preprocess class and selects the best variant.

For example, you could create a Polynomial class, which provides a generator that raises the dataset to a higher-order polynomial. The Pipeline would then iterate over these and train a model on each one, testing which has the best performance. Another Preprocess subclass could yield two versions of the same text dataset, one where the text is stemmed and another where it is not.

Multiple Pipelines can (and probably should) be chained together to evaluate a series of steps. The Pipeline(s) then apply the tranformation steps and the resultant data is then passed to a model.

Step 3 is represented by the Ensemble classes (ClassifierEnsemble and RegressionEnsemble). You can think of each model as a hyper parameter for the ensemble. Then the ensemble can be cross validated, tuning the voting weight for each model, and each model is cross validated.

When using the Ensemble, you are not limited to models provided by the FML framework. Any model with the correct API can be used. I intentionally based this off of scikit-learn to make it easy to use those out of the box. However, if you want to use another model that doesn't follow this API, there is information below on how to do that.

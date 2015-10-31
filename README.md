# Frankenstein Machine Learning (FML)

This is a machine learning framework to simplify developing predictive models.
This was originally intended to automate Kaggle competitions, but it can be used more generally for any modeling task.

# Setup

Currently dependent on sklearn.

# Tutorial

The idea is that much of the process of developing a predictive model can be automated if each step is cross validated.

The high-level steps to create any predictive model are

1. Preprocessing (e.g. stemming, lemmatizing, normalization)
2. Feature engineering (e.g. n-grams, higher-order polynomials)
3. Model selection and tuning (e.g. cross validating to tune hyper parameters)

The line betweeen preprocessing and feature engineering is always a bit blurry, and FML just handles them both the same way. Steps 1 and 2 are represented in FML by Preprocess classes. A Preprocessing class iterates over variations of a specific preprocessing step.

Subclasses of the Preprocess class are passed to an instance of the Pipeline class. A Pipeline object then iterates over the various transformations provided by the Preprocess class and selects the best variant.

For example, you could create a Polynomial class, which provides a generator that raises the dataset to a higher-order polynomial. The Pipeline would then iterate over these and train a model on each one, testing which has the best performance. Another Preprocess subclass could yield two versions of the same text dataset, one where the text is stemmed and another where it is not.

Multiple Pipelines can (and probably should) be chained together to evaluate a series of steps. The Pipeline(s) then apply the tranformation steps and the resultant data is then passed to a model.

Step 3 is represented by the Ensemble classes (ClassifierEnsemble and RegressionEnsemble). You can think of each model as a hyper parameter for the ensemble. Then the ensemble can be cross validated, tuning the voting weight for each model, and each model is cross validated.

When using the Ensemble, you are not limited to models provided by the FML framework. Any model with the correct API can be used. I intentionally based this off of scikit-learn to make it easy to use those out of the box. However, if you want to use another model that doesn't follow this API, I provide an example below on how to do that.

Overall, the idea is to cross validate a series of preprocessing/feature-engineering steps. Then apply the best steps to the dataset. Then cross validate an ensemble of multiple models, where the ensemble selects which models to use or discard.




## Version 1.0

### MISC
1. Fork class to send data to multiple models
2. Make Pipeline instances embeddable/chainable
3. import logger rather than assigning it to each class
4. requirements.txt
5. make .createdatasets() a shared function across the whole app
6. make a score function shared across the whole app

### Feature: Ensemble with configurable model (DONE)
1. Ensemble class that takes a set of models to CV and another model to vote with their outputs

### Feature: Config/settings file (DONE)
1. hyperparams should be configurable from this file

### Feature: Automate feature engineering
1. given data and targets, output function to generate features
2. see if it does better with higher order polynomial
3. see if it does better with tensor product
4. see if it does better with k-means clustering as input
5. see if it does better with PCA
6. normalize the input data
7. distribute input data with 0 mean and unit variance of 1
8. FeatureValidator class, which accepts a dataset,
   an array of preprocessing functions to apply to it, and an algorithm to CV it with

### Feature: More sklearn models
1. automate hyper param generator
2. linear models
3. RandomForest
4. naive bayes
5. kNN
6. SVM
7. xgboost




## post version 1.0

### Feature: neural nets

### Feature: code generation

### Feature: other libraries

# autoML
Data Science/Machine Learning Framework

# MVP

## Feature: Automate feature engineering
Requirements:
1. given data and targets, output function to generate features
2. see if it does better with higher order polynomial
3. see if it does better with tensor product
4. see if it does better with k-means clustering as input
5. see if it does better with PCA
6. normalize the input data
7. distribute input data with 0 mean and unit variance of 1
8. FeatureValidator class, which accepts a dataset,
   an array of preprocessing functions to apply to it, and an algorithm to CV it with

## Feature: More sklearn models
1. automate hyper param generator
2. linear models
3. RandomForest
4. naive bayes
5. kNN
6. SVM

## Feature: Ensemble with configurable model
1. Ensemble class that takes a set of models to CV and another model to vote with their outputs

## Feature: Controller
Requirements:
1. Accept a list of preprocessing steps to validate
2. Accept a list of models to validate

## Feature: Config/settings file
1. hyperparams should be configurable from this file

## Feature: requirements.txt
1. put all requirments into this file


# post version 1.0

## Feature: neural nets

## Feature: code generation

## Feature: other libraries

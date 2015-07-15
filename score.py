from sklearn.metrics import accuracy_score, mean_absolute_error

def score(preds, targets):
    return accuracy_score(targets, preds)

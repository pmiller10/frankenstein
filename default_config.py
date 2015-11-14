from sklearn.metrics import accuracy_score

class DefaultConfig():


    class Polynomial():

        START = 2
        STOP = 2


    @classmethod
    def loss(cls, preds, targets):
        #return accuracy_score(targets, preds)
        return sum([(p-t)**2 for p,t in zip(preds, targets)])

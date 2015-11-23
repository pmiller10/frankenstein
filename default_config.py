from constants import Objective

class DefaultConfig():


    # number of variations to cross validate per model
    epochs = 10

    # models should minimize or maximize the loss function
    objective = Objective.MINIMIZE


    class Polynomial():

        START = 2
        STOP = 5


    @classmethod
    def loss(cls, preds, targets):
        return sum([(p-t)**2 for p,t in zip(preds, targets)])

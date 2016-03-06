import lib
from _globals import Config
from sklearn import preprocessing



class Preprocess(object):

    def transform(self, data):
        """
        Method to be overridden in subclasses.
        Should accept a dataset.
        Should return the modified dataset.
        """
        return NotImplementedError



class Polynomial(Preprocess):


    def transform(self, dataset, exponent):
        return lib.polynomial(dataset, exponent)


    def each_transformation(self, dataset):
        for exponent in range(Config.Polynomial.START, Config.Polynomial.STOP):
            yield self.transform(dataset, exponent), {'exponent': exponent}



class Scale(Preprocess):


    def transform(self, dataset, scale=True):
        # Define the scale param even though it's not used.
        # That way it's consistent with other Preprocess subclasses.
        return preprocessing.scale(dataset)


    def each_transformation(self, dataset):
        for i in range(1):
            yield self.transform(dataset), {'scale': True}



class Norm(Preprocess):


    def transform(self, dataset, norm):
        return preprocessing.normalize(dataset, norm=norm)


    def each_transformation(self, dataset):
        norms = ['l1', 'l2']
        for n in norms:
            yield self.transform(dataset, norm=n), {'norm': n}

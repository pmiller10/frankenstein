import lib
from _globals import Config



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
        return lib.scale(dataset)


    def each_transformation(self, dataset):
        for i in range(1):
            yield self.transform(dataset), {'scale': True}



class Norm(Preprocess):


    def transform(self, dataset, normalize=True):
        return lib.norm(dataset)


    def each_transformation(self, dataset):
        for i in range(1):
            yield self.transform(dataset), {'normalize': True}

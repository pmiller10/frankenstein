import lib
import config

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


    def each_transformation(self, dataset, _):
        for exponent in range(config.Polynomial.START, config.Polynomial.STOP):
            yield self.transform(dataset, exponent), {'exponent': exponent}

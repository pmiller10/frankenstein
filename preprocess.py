class Preprocess(object):

    def transform(self, data):
        """
        Method to be overridden in subclasses.
        Should accept a dataset.
        Should return a generator of modified data.
        """
        return NotImplementedError



class Polynomial(Preprocess):

    pass

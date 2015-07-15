from sklearn import datasets



class Data():


    @classmethod
    def train(cls):
        iris = datasets.load_iris()
        return list(iris.data)[::2], list(iris.target)[::2]


    @classmethod
    def test(cls):
        iris = datasets.load_iris()
        return list(iris.data)[1::2], list(iris.target)[1::2]

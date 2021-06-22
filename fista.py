__author__ = "Ahmad Mohammad"


class Fista:

    """
    Fast iterative shrinking/thresholding algorithm
    """

    def __init__(self, lambda_=.5, iterations=1000):
        self.lambda_ = lambda_
        self.iterations = iterations

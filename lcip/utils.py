
class Simple_P_wrapper:
    def __init__(self, P, Pinv):
        self.P = P
        self.Pinv = Pinv
    def __call__(self, x):
        return self.P(x)
    def transform(self, x):
        return self.P.transform(x)
    # with keywrod argumentscon
    def inverse_transform(self, x, **kwargs):
        return self.Pinv.transform(x, **kwargs)

    def fit(self, x, x2d=None, **kwargs):
        if x2d is None:
            self.X2d = self.P.fit_transform(x).astype('float32')
        else:
            self.X2d = x2d
        self.Pinv.fit(self.X2d, x, **kwargs)
        return self
    
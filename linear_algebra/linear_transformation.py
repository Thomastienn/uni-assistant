from linear_algebra.matrix import Matrix


class LinearTransformation:
    # func is the transformation function that you define
    # it should have 1 argument which is type Matrix (should be a vector)
    # then transform into another Matrix (should be a vector too)

    # RN: the dimension of the input vector
    # RM: the dimension of the output vector
    def __init__(self, RN, RM, func):
        self.RN = RN
        self.RM = RM
        self.func = func
        # TODO
        # if (not self.is_linear()):
        #     assert False, "This is not a linear transformation"

    def get_transform_mat(self):
        A_mn = Matrix(a=[[] for _ in range(self.RM)])
        for i, standard_basis in enumerate(LinearTransformation.get_standard_basis(self.RN)):
            T_e = self.transform(standard_basis)
            A_mn = A_mn.concat(T_e)
        return A_mn

    def transform(self, vector) -> Matrix:
        if len(vector.a) != self.RN or not vector.is_vector():
            assert False, "Invalid vector"
        return self.func(vector)

    @staticmethod
    def get_standard_basis(n):
        for i in range(n):
            yield Matrix(a=[[0]*i + [1] + [0]*(n-i-1)]).T()

    # TODO
    def is_linear(self):
        zerov = self.zero_vec(len(self.args))
        if (self.func(zerov) != zerov):
            return False

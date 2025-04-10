from matrix import Matrix


class Transformation:
    # func is the transformation function that you define
    # it should have 1 argument which is type Matrix (should be a vector)
    # then transform into another Matrix (should be a vector too)

    # RN: the dimension of the input vector
    # RM: the dimension of the output vector
    def __init__(self, RN, RM, func):
        self.RN = RN
        self.RM = RM
        self.func = func

    def get_transform_mat(self, RN, RM):
        A_mn = [[-1] * RM for _ in range(RN)]
        for i, standard_basis in enumerate(self.get_standard_basis(len(self.args))):
            T_e = self.transform(standard_basis)
            for j in range(RM):
                A_mn[i][j] = T_e.args[j]
        return Matrix(a=A_mn)

    def transform(self, vector):
        if len(vector) != self.RN or not vector.is_vector():
            assert False, "Invalid vector"
        return self.func(vector)

    def zero_vec(self, n):
        return Matrix(a=([[self.t("0")] for _ in range(n)]), t=self.t)

    def get_standard_basis(self, n):
        for i in range(n):
            yield Matrix(a=[0]*i + [1] + [0]*(n-i-1))

    # TODO
    def is_linear(self):
        zerov = self.zero_vec(len(self.args))
        if (self.func(zerov) != zerov):
            return False

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
        self.inv_func = None
        # TODO
        # if (not self.is_linear()):
        #     assert False, "This is not a linear transformation"

    def get_transform_mat(self):
        A_mn = Matrix(a=[[] for _ in range(self.RM)])
        for i, standard_basis in enumerate(LinearTransformation.get_standard_basis(self.RN)):
            T_e = self.transform(standard_basis)
            A_mn.concat(T_e, in_place=True)
        return A_mn

    # B: list of basis from Rn
    # D: list of basis from Rm
    # return tranformed matrix A from Rn to Rm (using basis from each R)
    def get_transform_ADB(self, B: list[Matrix], D: list[Matrix]):
        A_db = Matrix(a=[[] for _ in range(len(D))])
        for b_basis in B:
            transform_basis = self.transform(b_basis)
            A_db.concat(transform_basis.cB(D), in_place=True)
        return A_db

    def transform(self, vector) -> Matrix:
        if len(vector.a) != self.RN or not vector.is_vector():
            assert False, "Invalid vector"
        return self.func(vector)

    @staticmethod
    def get_standard_basis(n):
        for i in range(n):
            yield Matrix(a=[[0]*i + [1] + [0]*(n-i-1)]).T()

    def get_inverse_transform_func(self):
        if self.inv_func is not None:
            return self.inv_func
        transform_a = self.get_transform_mat()
        if transform_a.det() == 0:
            assert False, "Tranform cannot be inversed"

        inverse_ta = transform_a.inv()

        def new_func(vector):
            new_a = [[0] for _ in range(len(vector.a))]
            for i in range(len(inverse_ta.a)):
                for j in range(len(inverse_ta.a[0])):
                    new_a[i][-1] += (inverse_ta.a[i][j]*vector.vR(j))
                    print(new_a)
                    print(inverse_ta.a[i][j], vector.vR(j))
            return Matrix(a=new_a)

        self.inv_func = new_func
        return new_func

    def inv_transform(self, vector):
        if len(vector.a) != self.RM or not vector.is_vector():
            assert False, "Invalid vector"
        return self.get_inverse_transform_func()(vector)

    # TODO unfinished
    def is_linear(self):
        zerov = Matrix.zero_vec(len(self.RN))
        if (self.func(zerov) != zerov):
            return False
        return True

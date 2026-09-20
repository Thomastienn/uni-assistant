from linear_algebra import _matrix_algebra as _algebra
from linear_algebra import _matrix_spectral as _spectral
from linear_algebra import _matrix_vectors as _vectors


class Matrix:
    def __init__(self, a=None, t=eval):
        self.t = t
        if a is None:
            self._get()
        else:
            self.a = a

    def __iter__(self):
        return iter(self.a)

    def __len__(self):
        return len(self.a)

    def __getitem__(self, index):
        return self.a[index]

    def __eq__(self, other: "Matrix"):
        n, m = len(self), len(self[0])
        k, l = len(other), len(other[0])
        if n != k or m != l:
            return False
        for i in range(n):
            for j in range(m):
                if self[i][j] != other[i][j]:
                    return False
        return True

    def __mul__(self, bmat):
        ans = None
        if isinstance(bmat, Matrix):
            if len(self[0]) != len(bmat):
                assert False, "not matching size"
            ans = [[self.t("0")]*len(bmat[0]) for _ in range(len(self))]
            for i in range(len(self)):
                for j in range(len(bmat[0])):
                    for k in range(len(bmat)):
                        ans[i][j] += self[i][k]*bmat[k][j]
        else:
            ans = [[0]*len(self[0]) for _ in range(len(self))]
            for i in range(len(self)):
                for j in range(len(self[0])):
                    ans[i][j] = self[i][j]*bmat
        return Matrix(ans, self.t)

    def __rmul__(self, scalar):
        return self * scalar

    def __add__(self, bmat: "Matrix"):
        return self.add(bmat, 1)

    def __sub__(self, bmat: "Matrix"):
        return self.add(bmat, -1)

    def __pow__(self, n):
        amat = self._copyMat()
        iden = Matrix.imat(len(self), self.t)
        while n:
            if n & 1:
                iden *= amat
            amat *= amat
            n >>= 1
        return iden

    def add(self, bmat: "Matrix", delta=1):
        b = bmat.a
        if len(self) != len(b) or len(self[0]) != len(b[0]):
            return None
        ans = [[0]*len(self[0]) for _ in range(len(self))]
        for i in range(len(self)):
            for j in range(len(self[0])):
                ans[i][j] = (self[i][j] + (b[i][j]*delta))

        return Matrix(ans, self.t)

    def _copyArr(self):
        return [row[:] for row in self]

    def _copyMat(self):
        return Matrix(self._copyArr(), t=self.t)

    def removeRow(self, row):
        self.a.pop(row)

    def removeCol(self, col):
        n = len(self)
        for i in range(n):
            self[i].pop(col)

    def assignRow(self, bdelta, brow, adelta, arow, mainR):
        new = self._copyMat()
        for c in range(len(self[0])):
            new[mainR][c] = self[brow][c]*bdelta + self[arow][c]*adelta
        return new

    def changeRow(self, arow, brow, other):
        new_a = self._copyArr()
        other_a = other._copyArr()
        new_a[arow] = other_a[brow]
        return Matrix(new_a, t=self.t)

    def changeCol(self, acol, bcol, other):
        new_a = self._copyArr()
        other_a = other._copyArr()
        for i in range(len(self)):
            new_a[i][acol] = other_a[i][bcol]
        return Matrix(new_a, t=self.t)

    def swapRow(self, arow, brow):
        new_a = self._copyArr()
        new_a[arow], new_a[brow] = new_a[brow], new_a[arow]
        return Matrix(new_a, t=self.t)

    def T(self):
        return Matrix([list(r) for r in zip(*self)], self.t)

    def rot90(self):
        return Matrix([r[::-1] for r in self.T()], self.t)

    def concat(self, bmat, in_place=False):
        if len(bmat) != len(self):
            assert False, "not matching size"

        if in_place:
            new = self.a
        else:
            new = self._copyArr()
        for i in range(len(new)):
            new[i] += bmat[i]

        if not in_place:
            return Matrix(new, t=self.t)

    @staticmethod
    def im(n, t=eval):
        return [[t("1") if i == j else t("0") for j in range(n)] for i in range(n)]  # noqa

    @staticmethod
    def imat(n, t=eval):
        return Matrix(Matrix.im(n, t), t=t)

    @staticmethod
    def zero_vec(n, t=eval):
        return Matrix(([[t("0")] for _ in range(n)]), t=t)

    @staticmethod
    def mvec(*elements, t=eval):
        return Matrix([[x] for x in elements], t=t)

    def _get(self):
        self.a = []
        n = int(input())
        for _ in range(n):
            self.a.append(list(map(self.t, input().split())))

    def __str__(self):
        s = ""
        for r in self:
            s += " ".join(map(str, r)) + "\n"
        return s

    def show(self):
        for r in self:
            print(*r)
        print()

    def print_l(self):
        print(str(self.a).replace(" ", ""))

    def det2d(self):
        return _algebra.det2d(self)

    def det3d(self):
        return _algebra.det3d(self)

    def sign_cof(self, row, col):
        return _algebra.sign_cof(self, row, col)

    def minor(self, row, col):
        return _algebra.minor(self, row, col)

    def cof(self, row, col):
        return _algebra.cof(self, row, col)

    def minorMat(self):
        return _algebra.minorMat(self)

    def cofMat(self):
        return _algebra.cofMat(self)

    def adj(self):
        return _algebra.adj(self)

    def det(self):
        return _algebra.det(self)

    def inv_MIA(self):
        return _algebra.inv_MIA(self)

    def inv(self):
        return _algebra.inv(self)

    def inv2d(self):
        return _algebra.inv2d(self)

    @staticmethod
    def isinv(a: "Matrix", b: "Matrix"):
        return _algebra.isinv(a, b)

    def solve(self, b):
        return _algebra.solve(self, b)

    def solveSelf(self):
        return _algebra.solveSelf(self)

    def isrref(self, arr=None):
        return _algebra.isrref(self, arr)

    def rref(self, tol=1e-12):
        return _algebra.rref(self, tol)

    def col_space(self, tol=1e-12):
        return _algebra.col_space(self, tol)

    def row_space(self, tol=1e-12):
        return _algebra.row_space(self, tol)

    def is_vector(self):
        return _vectors.is_vector(self)

    def vR(self, pos):
        return _vectors.vR(self, pos)

    def dot(self, other: "Matrix"):
        return _vectors.dot(self, other)

    def cross(self, other: "Matrix"):
        return _vectors.cross(self, other)

    def cB(self, basis: list["Matrix"]):
        return _vectors.cB(self, basis)

    def in_span(self, basis: list["Matrix"]):
        return _vectors.in_span(self, basis)

    def cA(self):
        return _spectral.cA(self)

    def eigen_vals(self):
        return _spectral.eigen_vals(self)

    def is_similar(self, other):
        return _spectral.is_similar(self, other)

    def eigen_vec(self, eigen_val):
        return _spectral.eigen_vec(self, eigen_val)

    def algebraic_multiplicity(self, eigen_val):
        return _spectral.algebraic_multiplicity(self, eigen_val)

    def geometric_multiplicity(self, eigen_val):
        return _spectral.geometric_multiplicity(self, eigen_val)

    def is_diagnolizable(self):
        return _spectral.is_diagnolizable(self)

    def diag(self):
        return _spectral.diag(self)

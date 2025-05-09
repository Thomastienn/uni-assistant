from fractions import Fraction
from linear_algebra.polynomial import Poly
import numpy as np


class Matrix:
    def __init__(self, t=eval, a=None):
        self.t = t
        if a is None:
            self._get()
        else:
            self.a = a

    def __eq__(self, other: "Matrix"):
        n, m = len(self.a), len(self.a[0])
        k, l = len(other.a), len(other.a[0])
        if n != k or m != l:
            return False
        for i in range(n):
            for j in range(m):
                if self.a[i][j] != other.a[i][j]:
                    return False
        return True

    def __mul__(self, bmat):
        ans = None
        if isinstance(bmat, Matrix):
            if len(self.a[0]) != len(bmat.a):
                assert False, "not matching size"
            ans = [[self.t("0")]*len(bmat.a[0]) for _ in range(len(self.a))]
            for i in range(len(self.a)):
                for j in range(len(bmat.a[0])):
                    for k in range(len(bmat.a)):
                        ans[i][j] += self.a[i][k]*bmat.a[k][j]
        else:
            ans = [[0]*len(self.a[0]) for _ in range(len(self.a))]
            for i in range(len(self.a)):
                for j in range(len(self.a[0])):
                    ans[i][j] = self.a[i][j]*bmat
        return Matrix(self.t, ans)

    def __add__(self, bmat: "Matrix"):
        return self.add(bmat, 1)

    def __sub__(self, bmat: "Matrix"):
        return self.add(bmat, -1)

    def __pow__(self, n):
        amat = self._copyMat()
        iden = Matrix(a=Matrix.im(len(self.a), self.t))
        while n:
            if n & 1:
                iden *= amat
            amat *= amat
            n >>= 1
        return iden

    def removeRow(self, row):
        self.a.pop(row)

    def removeCol(self, col):
        n = len(self.a)
        for i in range(n):
            self.a[i].pop(col)

    def det2d(self):
        n, m = len(self.a), len(self.a[0])
        assert (n == 2 and m == 2)

        return self.a[0][0]*self.a[1][1] - self.a[0][1]*self.a[1][0]

    # Sarrus's Rule
    def det3d(self):
        assert len(self.a) == len(self.a[0]) == 3
        s = self.t("0")
        for off in range(3):
            pro1 = self.t("1")
            pro2 = self.t("1")
            for i in range(3):
                pro1 *= self.a[i][(i+off) % 3]
                pro2 *= self.a[i][(off-i) % 3]
            s += pro1 - pro2
        return s

    def sign_cof(self, row, col):
        return -1 if (row+col) & 1 else 1

    def minor(self, row, col):
        n, m = len(self.a), len(self.a[0])
        new_a = []
        for i in range(n):
            for j in range(m):
                if i == row or j == col:
                    continue
                if not new_a or len(new_a[-1]) == m-1:
                    new_a.append([])
                new_a[-1].append(self.a[i][j])
        new_mat = Matrix(a=new_a)
        return new_mat.det()

    def cof(self, row, col):
        return self.sign_cof(row, col)*self.minor(row, col)

    def minorMat(self):
        new = [[0]*len(self.a[0]) for _ in range(len(self.a))]
        for i in range(len(self.a)):
            for j in range(len(self.a[0])):
                new[i][j] = self.minor(i, j)
        return Matrix(a=new)

    def cofMat(self):
        new = [[0]*len(self.a[0]) for _ in range(len(self.a))]
        for i in range(len(self.a)):
            for j in range(len(self.a[0])):
                new[i][j] = self.cof(i, j)
        return Matrix(a=new)

    def adj(self):
        return self.cofMat().T()

    # Naive method
    # def det(self):
        # n,m = len(self.a), len(self.a[0])
        # assert n == m
        # if n == 2:
        # return self.det2d()
        # ans = 0
        # # always first row
        # for i in range(m):
        # ans += self.a[0][i] * self.cof(0,i)
        # return ans

    # Using adjugate
    def det(self):
        if len(self.a) == 0:
            return 1
        n, m = len(self.a), len(self.a[0])
        assert n == m
        if n == 2:
            return self.det2d()
        if n == 3:
            return self.det3d()
        return (self*self.adj()).a[0][0]

    def _copyArr(self):
        new = [[-1]*len(self.a[0]) for _ in range(len(self.a))]
        for i in range(len(self.a)):
            for j in range(len(self.a[0])):
                new[i][j] = self.a[i][j]
        return new

    def _copyMat(self):
        return Matrix(a=self._copyArr())

    # Helpful in finding rref
    # by applying row operations
    def assignRow(self, bdelta, brow, adelta, arow, mainR):
        new = self._copyMat()
        for c in range(len(self.a[0])):
            new.a[mainR][c] = self.a[brow][c]*bdelta + self.a[arow][c]*adelta
        return new

    # Using cramer's rule
    def solve(self, b):
        detA = self.det()
        new_a = [Fraction(self.changeCol(i, 0, b).det(), detA) for i in range(len(self.a))]  # noqa
        return Matrix(a=[new_a])

    def changeRow(self, arow, brow, other):
        new_a = self._copyArr()
        other_a = other._copyArr()
        new_a[arow] = other_a[brow]
        return Matrix(a=new_a)

    def changeCol(self, acol, bcol, other):
        new_a = self._copyArr()
        other_a = other._copyArr()
        for i in range(len(self.a)):
            new_a[i][acol] = other_a[i][bcol]
        return Matrix(a=new_a)

    def swapRow(self, arow, brow):
        new_a = self._copyArr()
        new_a[arow], new_a[brow] = new_a[brow], new_a[arow]
        return Matrix(a=new_a)

    def _get(self):
        self.a = []
        n = int(input())
        for _ in range(n):
            self.a.append(list(map(self.t, input().split())))

    def add(self, bmat: "Matrix", delta=1):
        b = bmat.a
        if len(self.a) != len(b) or len(self.a[0]) != len(b[0]):
            return None
        ans = [[0]*len(self.a[0]) for _ in range(len(self.a))]
        for i in range(len(self.a)):
            for j in range(len(self.a[0])):
                ans[i][j] = (self.a[i][j] + (b[i][j]*delta))

        return Matrix(self.t, ans)

    # Transpose
    def T(self):
        return Matrix(self.t, [list(r) for r in zip(*self.a)])

    # LLM WORK
    # Matrix Inversion Algorithm
    def inv_MIA(self):
        A = [row[:] for row in self.a]
        I_MAT = Matrix.im(len(self.a), self.t)
        for i in range(len(self.a)):
            pivot = A[i][i]
            if pivot == 0:
                for j in range(i + 1, len(self.a)):
                    if A[j][i] != 0:
                        A[i], A[j] = A[j], A[i]
                        I_MAT[i], I_MAT[j] = I_MAT[j], I_MAT[i]
                        pivot = A[i][i]
                        break
            if pivot == 0:
                raise ValueError("Matrix is singular and cannot be inverted.")
            for j in range(len(self.a)):
                A[i][j] /= pivot
                I_MAT[i][j] /= pivot
            for j in range(len(self.a)):
                if i != j:
                    factor = A[j][i]
                    for k in range(len(self.a)):
                        A[j][k] -= factor * A[i][k]
                        I_MAT[j][k] -= factor * I_MAT[i][k]
        return Matrix(type(I_MAT[0][0]), I_MAT)

    # Inverse by determinant and adjungate
    def inv(self):
        if len(self.a) == 0:
            return Matrix(a=[])
        assert len(self.a) == len(self.a[0]), "Must be a square"
        if len(self.a) == 1:
            assert self.a[0][0] != 0, "No inverse"
            return Matrix(a=[[Fraction(1, self.a[0][0])]])
        if len(self.a) == 2:
            return self.inv2d()
        new_mat = self._copyMat()
        return new_mat.adj()*(1/new_mat.det())

    def inv2d(self):
        new_mat = Matrix(a=[[self.a[1][1], -self.a[0][1]],
                         [-self.a[1][0], self.a[0][0]]])
        return new_mat*(1/(self.a[0][0]*self.a[1][1] - self.a[0][1]*self.a[1][0]))  # noqa

    def rot90(self):
        return Matrix(self.t, [r[::-1] for r in self.T(self.a)])

    @staticmethod
    def isinv(a: "Matrix", b: "Matrix"):
        return a*b == b*a

    # Display matrix
    def show(self):
        for r in self.a:
            print(*r)
        print()

    # Display as an array
    def print_l(self):
        print(str(self.a).replace(" ", ""))

    def _rowNonZero(self, rows):
        for i in range(len(rows)):
            if rows[i] != 0:
                return i
        return len(rows)

    def _rearrange(self, arr):
        arr.sort(key=lambda r: self._rowNonZero(r))

    # is Row Reduced Echelon Form
    def isrref(self, arr=None):
        if arr is None:
            arr = self._copyArr()
        prev = None
        for r in range(len(arr)):
            i = self._rowNonZero(arr[r])

            # FULL OF ZEROS
            if i == len(arr[r]) and r != len(arr)-1:
                return False

            # BELOW AND ABOVE MUST BE 0 TOO
            for r1 in range(r+1, len(arr)):
                if arr[r1][i] != 0:
                    return False
            for r1 in range(r):
                if arr[r1][i] != 0:
                    return False

            # THE LAST ONE MUST BE MORE LEFT
            if prev is not None and i <= prev:
                return False
            prev = i

        return True

    # Concat sideways
    def concat(self, bmat, in_place=False):
        if len(bmat.a) != len(self.a):
            assert False, "not matching size"

        if in_place:
            new = self.a
        else:
            new = self._copyArr()
        for i in range(len(new)):
            new[i] += bmat.a[i]

        if not in_place:
            return Matrix(a=new)

    # LLM WORK
    # TODO rework this too
    def rref(self):
        A = self._copyArr()
        rows, cols = len(A), len(A[0])
        r = 0  # Row index

        for c in range(cols):
            # Find the row with the largest absolute value in column c
            pivot_row = max(range(r, rows), key=lambda i: abs(
                A[i][c]), default=None)
            if pivot_row is None or A[pivot_row][c] == 0:
                continue  # Skip if column is all zeros

            # Swap the current row with the pivot row
            A[r], A[pivot_row] = A[pivot_row], A[r]

            # Normalize the pivot row (make leading coefficient 1)
            pivot = A[r][c]
            A[r] = [val / pivot for val in A[r]]

            # Eliminate all other entries in the current column
            for i in range(rows):
                if i != r:
                    factor = A[i][c]
                    A[i] = [A[i][j] - factor * A[r][j] for j in range(cols)]

            r += 1  # Move to the next row

        return Matrix(a=A, t=self.t)

    # identity matrix as an array
    @staticmethod
    def im(n, t=eval):
        return [[t("1") if i == j else t("0") for j in range(n)] for i in range(n)]  # noqa

    # identity matrix
    @staticmethod
    def imat(n, t=eval):
        return Matrix(a=Matrix.im(n, t))

    @staticmethod
    def zero_vec(n, t):
        return Matrix(a=([[t("0")] for _ in range(n)]), t=t)

    def is_vector(self):
        for i in range(len(self.a)):
            if len(self.a[i]) != 1:
                return False
        return True

    # Get which row of the column vector
    def vR(self, pos):
        if not self.is_vector():
            assert False, "not a vector, cannot use"
        return self.a[pos][0]

    # Characteristic Polynomial
    def cA(self):
        if len(self.a) != len(self.a[0]):
            assert False, "need to be nxn"
        lambdaa = Poly("x")
        mat = (Matrix.imat(len(self.a), self.t) * lambdaa) - self

        return mat.det()

    # basis: list of basis vectors
    # Give a coordinate vector base on the basis
    def cB(self, basis: list["Matrix"]):
        solve_mat = Matrix(a=[[] * len(basis)
                           for _ in range(len(basis[0].a))], t=self.t)
        for basis_vec in basis:
            solve_mat.concat(basis_vec, in_place=True)

        return solve_mat.solve(self).T()

    # Create a vector
    @staticmethod
    def mvector(elements: list, t=eval):
        return Matrix(a=[[x] for x in elements], t=t)

    # Eigenvalues
    def eigen_vals(self):
        equal = self.cA()
        roots = np.roots(equal.coef[::-1])
        ret = [round(float(x.real), 3) if abs(
            x.imag) < 1e-5 else x for x in roots]
        return ret

    # chek if they are similar
    # TODO in progress
    # not fully functional, will add jordan form check
    def is_similar(self, other):
        n, m = len(self.a), len(self.a[0])
        k, l = len(other.a), len(other.a[0])
        if n != k or m != l:
            return False
        if self.det() != other.det():
            return False
        return sorted(self.eigen_vals()) == sorted(other.eigen_vals())

    # This can only give you a view of rref
    # you need to look at the augmented matrix and get the vector
    # return rref matrix represent the eigen vector of the corresponding eigen value
    def eigen_vec(self, eigen_val):
        if len(self.a) != len(self.a[0]):
            assert False, "need to be nxn"

        solve_mat = Matrix.imat(len(self.a), self.t)
        solve_mat = solve_mat * eigen_val - self
        zero_vec = Matrix.zero_vec(len(self.a), self.t)

        res_mat = solve_mat.concat(zero_vec)
        return res_mat.rref()

    def algebraic_multiplicity(self, eigen_val):
        return self.eigen_vals().count(eigen_val)

    def geometric_multiplicity(self, eigen_val):
        cnt = 0
        for row in self.eigen_vec(eigen_val).a:
            zeros = 0
            for c in row:
                if c == 0:
                    zeros += 1
                    continue
                    if isinstance(c, Poly) and c == Poly("0"):
                        zeros += 1
            cnt += zeros == len(row)

        return cnt

    def is_diagnolizable(self):
        for eigen_val in self.eigen_vals():
            alg_mult = self.algebraic_multiplicity(eigen_val)
            geo_mult = self.geometric_multiplicity(eigen_val)
            if alg_mult != geo_mult:
                return False
        return True

    # The diagonal matrix of a
    def diag(self):
        if not self.is_diagnolizable():
            assert False, "not diagnolizable"
        eigenval = self.eigen_vals()
        new_a = Matrix.imat(len(self.a), self.t)
        for i in range(len(self.a)):
            new_a[i][i] = eigenval[i]
        return Matrix(a=new_a, t=self.t)

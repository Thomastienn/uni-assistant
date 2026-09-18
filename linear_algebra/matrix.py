from fractions import Fraction
from linear_algebra.polynomial import Poly
import numpy as np


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
        iden = Matrix(Matrix.im(len(self), self.t))
        while n:
            if n & 1:
                iden *= amat
            amat *= amat
            n >>= 1
        return iden

    def removeRow(self, row):
        self.a.pop(row)

    def removeCol(self, col):
        n = len(self)
        for i in range(n):
            self[i].pop(col)

    def det2d(self):
        n, m = len(self), len(self[0])
        assert (n == 2 and m == 2)

        return self[0][0]*self[1][1] - self[0][1]*self[1][0]

    # Sarrus's Rule
    def det3d(self):
        assert len(self) == len(self[0]) == 3
        s = self.t("0")
        for off in range(3):
            pro1 = self.t("1")
            pro2 = self.t("1")
            for i in range(3):
                pro1 *= self[i][(i+off) % 3]
                pro2 *= self[i][(off-i) % 3]
            s += pro1 - pro2
        return s

    def sign_cof(self, row, col):
        return -1 if (row+col) & 1 else 1

    def minor(self, row, col):
        n, m = len(self), len(self[0])
        new_a = []
        for i in range(n):
            for j in range(m):
                if i == row or j == col:
                    continue
                if not new_a or len(new_a[-1]) == m-1:
                    new_a.append([])
                new_a[-1].append(self[i][j])
        new_mat = Matrix(new_a)
        return new_mat.det()

    def cof(self, row, col):
        return self.sign_cof(row, col)*self.minor(row, col)

    def minorMat(self):
        new = [[0]*len(self[0]) for _ in range(len(self))]
        for i in range(len(self)):
            for j in range(len(self[0])):
                new[i][j] = self.minor(i, j)
        return Matrix(new)

    def cofMat(self):
        new = [[0]*len(self[0]) for _ in range(len(self))]
        for i in range(len(self)):
            for j in range(len(self[0])):
                new[i][j] = self.cof(i, j)
        return Matrix(new)

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
        if len(self) == 0:
            return 1
        n, m = len(self), len(self[0])
        assert n == m
        if n == 2:
            return self.det2d()
        if n == 3:
            return self.det3d()
        return (self*self.adj())[0][0]

    def _copyArr(self):
        new = [[-1]*len(self[0]) for _ in range(len(self))]
        for i in range(len(self)):
            for j in range(len(self[0])):
                new[i][j] = self[i][j]
        return new

    def _copyMat(self):
        return Matrix(self._copyArr())

    # Helpful in finding rref
    # by applying row operations
    def assignRow(self, bdelta, brow, adelta, arow, mainR):
        new = self._copyMat()
        for c in range(len(self[0])):
            new[mainR][c] = self[brow][c]*bdelta + self[arow][c]*adelta
        return new

    # Using cramer's rule
    def solve(self, b):
        detA = self.det()
        new_a = [Fraction(self.changeCol(i, 0, b).det(), detA) for i in range(len(self))]  # noqa
        return Matrix([new_a])

    def solveSelf(self):
        A = Matrix([row[:-1] for row in self], self.t)
        b = Matrix([[row[-1]] for row in self], self.t)
        return A.solve(b)

    def changeRow(self, arow, brow, other):
        new_a = self._copyArr()
        other_a = other._copyArr()
        new_a[arow] = other_a[brow]
        return Matrix(new_a)

    def changeCol(self, acol, bcol, other):
        new_a = self._copyArr()
        other_a = other._copyArr()
        for i in range(len(self)):
            new_a[i][acol] = other_a[i][bcol]
        return Matrix(new_a)

    def swapRow(self, arow, brow):
        new_a = self._copyArr()
        new_a[arow], new_a[brow] = new_a[brow], new_a[arow]
        return Matrix(new_a)

    def _get(self):
        self.a = []
        n = int(input())
        for _ in range(n):
            self.a.append(list(map(self.t, input().split())))

    def add(self, bmat: "Matrix", delta=1):
        b = bmat.a
        if len(self) != len(b) or len(self[0]) != len(b[0]):
            return None
        ans = [[0]*len(self[0]) for _ in range(len(self))]
        for i in range(len(self)):
            for j in range(len(self[0])):
                ans[i][j] = (self[i][j] + (b[i][j]*delta))

        return Matrix(ans, self.t)

    # Transpose
    def T(self):
        return Matrix([list(r) for r in zip(*self)], self.t)

    # LLM WORK
    # Matrix Inversion Algorithm
    def inv_MIA(self):
        A = [row[:] for row in self]
        I_MAT = Matrix.im(len(self), self.t)
        for i in range(len(self)):
            pivot = A[i][i]
            if pivot == 0:
                for j in range(i + 1, len(self)):
                    if A[j][i] != 0:
                        A[i], A[j] = A[j], A[i]
                        I_MAT[i], I_MAT[j] = I_MAT[j], I_MAT[i]
                        pivot = A[i][i]
                        break
            if pivot == 0:
                raise ValueError("Matrix is singular and cannot be inverted.")
            for j in range(len(self)):
                A[i][j] /= pivot
                I_MAT[i][j] /= pivot
            for j in range(len(self)):
                if i != j:
                    factor = A[j][i]
                    for k in range(len(self)):
                        A[j][k] -= factor * A[i][k]
                        I_MAT[j][k] -= factor * I_MAT[i][k]
        return Matrix(I_MAT, type(I_MAT[0][0]))

    # Inverse by determinant and adjungate
    def inv(self):
        if len(self) == 0:
            return Matrix([])
        assert len(self) == len(self[0]), "Must be a square"
        if len(self) == 1:
            assert self[0][0] != 0, "No inverse"
            return Matrix([[Fraction(1, self[0][0])]])
        if len(self) == 2:
            return self.inv2d()
        new_mat = self._copyMat()
        return new_mat.adj()*(1/new_mat.det())

    def inv2d(self):
        new_mat = Matrix([[self[1][1], -self[0][1]],
                         [-self[1][0], self[0][0]]])
        return new_mat*(1/(self[0][0]*self[1][1] - self[0][1]*self[1][0]))  # noqa

    def rot90(self):
        return Matrix([r[::-1] for r in self.T(self.a)], self.t)

    @staticmethod
    def isinv(a: "Matrix", b: "Matrix"):
        return a*b == b*a

    # Display matrix
    def show(self):
        for r in self:
            print(*r)
        print()
    
    def __str__(self):
        s = ""
        for r in self:
            s += " ".join(map(str, r)) + "\n"
        return s

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
            arr = self.a
        if not arr:
            return True
        if any(len(row) != len(arr[0]) for row in arr):
            return False

        previous_pivot = -1
        found_zero_row = False
        for row_index, row in enumerate(arr):
            pivot = self._rowNonZero(row)
            if pivot == len(row):
                found_zero_row = True
                continue
            if found_zero_row or pivot <= previous_pivot or row[pivot] != 1:
                return False
            if any(other[pivot] != 0 for i, other in enumerate(arr) if i != row_index):
                return False
            previous_pivot = pivot
        return True

    # Concat sideways
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
            return Matrix(new)

    def rref(self, tol=1e-12):
        if tol < 0:
            raise ValueError("Tolerance must be nonnegative.")
        A = [row[:] for row in self]
        if not A:
            return Matrix([], t=self.t)
        rows, cols = len(A), len(A[0])
        if any(len(row) != cols for row in A):
            raise ValueError("Matrix rows must have the same length.")

        # Keep integer and fraction input exact during division.
        exact = all(isinstance(value, (int, Fraction)) for row in A for value in row)
        if exact:
            A = [[Fraction(value) for value in row] for row in A]
        tolerance = 0 if exact else tol
        pivot_row = 0

        for col in range(cols):
            if pivot_row == rows:
                break
            best_row = max(range(pivot_row, rows), key=lambda i: abs(A[i][col]))
            if abs(A[best_row][col]) <= tolerance:
                for i in range(pivot_row, rows):
                    A[i][col] = 0
                continue

            A[pivot_row], A[best_row] = A[best_row], A[pivot_row]
            pivot = A[pivot_row][col]
            A[pivot_row] = [value / pivot for value in A[pivot_row]]
            A[pivot_row][col] = 1

            for i in range(rows):
                if i == pivot_row:
                    continue
                factor = A[i][col]
                A[i] = [value - factor * pivot_value
                        for value, pivot_value in zip(A[i], A[pivot_row])]
                A[i][col] = 0
            pivot_row += 1

        return Matrix(A, t=Fraction if exact else self.t)

    # identity matrix as an array
    @staticmethod
    def im(n, t=eval):
        return [[t("1") if i == j else t("0") for j in range(n)] for i in range(n)]  # noqa

    # identity matrix
    @staticmethod
    def imat(n, t=eval):
        return Matrix(Matrix.im(n, t))

    @staticmethod
    def zero_vec(n, t=eval):
        return Matrix(([[t("0")] for _ in range(n)]), t=t)

    def is_vector(self):
        for i in range(len(self)):
            if len(self[i]) != 1:
                return False
        return True

    # Get which row of the column vector
    def vR(self, pos):
        if not self.is_vector():
            assert False, "not a vector, cannot use"
        return self[pos][0]

    # Characteristic Polynomial
    def cA(self):
        if len(self) != len(self[0]):
            assert False, "need to be nxn"
        lambdaa = Poly("x")
        mat = (Matrix.imat(len(self), self.t) * lambdaa) - self

        return mat.det()

    # basis: list of basis vectors
    # Give a coordinate vector base on the basis
    def cB(self, basis: list["Matrix"]):
        solve_mat = Matrix([[] * len(basis)
                           for _ in range(len(basis[0]))], t=self.t)
        for basis_vec in basis:
            solve_mat.concat(basis_vec, in_place=True)

        return solve_mat.solve(self).T()

    # Create a vector
    @staticmethod
    def mvec(*elements, t=eval):
        return Matrix([[x] for x in elements], t=t)

    def dot(self, other: "Matrix"):
        if not self.is_vector() or not other.is_vector() or len(self) != len(other):
            raise ValueError("Dot product requires column vectors of the same size.")
        return sum(a[0] * b[0] for a, b in zip(self, other))

    def cross(self, other: "Matrix"):
        if not self.is_vector() or not other.is_vector() or len(self) != 3 or len(other) != 3:
            raise ValueError("Cross product requires two 3D column vectors.")
        x, y, z = (row[0] for row in self)
        other_x, other_y, other_z = (row[0] for row in other)
        return Matrix.mvec(
            y * other_z - z * other_y,
            z * other_x - x * other_z,
            x * other_y - y * other_x,
            t=self.t)

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
        n, m = len(self), len(self[0])
        k, l = len(other), len(other[0])
        if n != k or m != l:
            return False
        if self.det() != other.det():
            return False
        return sorted(self.eigen_vals()) == sorted(other.eigen_vals())

    # This can only give you a view of rref
    # you need to look at the augmented matrix and get the vector
    # return rref matrix represent the eigen vector of the corresponding eigen value
    def eigen_vec(self, eigen_val):
        if len(self) != len(self[0]):
            assert False, "need to be nxn"

        solve_mat = Matrix.imat(len(self), self.t)
        solve_mat = solve_mat * eigen_val - self
        zero_vec = Matrix.zero_vec(len(self), self.t)

        res_mat = solve_mat.concat(zero_vec)
        return res_mat.rref()

    def algebraic_multiplicity(self, eigen_val):
        return self.eigen_vals().count(eigen_val)

    def geometric_multiplicity(self, eigen_val):
        cnt = 0
        for row in self.eigen_vec(eigen_val):
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
        new_a = Matrix.imat(len(self), self.t)
        for i in range(len(self)):
            new_a[i][i] = eigenval[i]
        return Matrix(new_a, t=self.t)

    def in_span(self, basis: list["Matrix"]):
        if not self.is_vector():
            raise ValueError("Target must be a column vector.")
        for vector in basis:
            if not vector.is_vector() or len(vector) != len(self):
                raise ValueError("Basis vectors must be column vectors matching the target size.")

        if not basis or not self:
            return all(row[0] == 0 for row in self)

        augmented = basis[0]._copyMat()
        for vector in basis[1:]:
            augmented.concat(vector, in_place=True)
        augmented.concat(self, in_place=True)

        for row in augmented.rref():
            coefficients = row[:-1]
            target = row[-1]
            if all(value == 0 for value in coefficients) and target != 0:
                return False

        return True

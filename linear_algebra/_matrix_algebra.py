from fractions import Fraction

from linear_algebra import matrix as _matrix


def det2d(matrix):
    n, m = len(matrix), len(matrix[0])
    assert n == 2 and m == 2

    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def det3d(matrix):
    assert len(matrix) == len(matrix[0]) == 3
    s = matrix.t("0")
    for off in range(3):
        pro1 = matrix.t("1")
        pro2 = matrix.t("1")
        for i in range(3):
            pro1 *= matrix[i][(i + off) % 3]
            pro2 *= matrix[i][(off - i) % 3]
        s += pro1 - pro2
    return s


def sign_cof(matrix, row, col):
    return -1 if (row + col) & 1 else 1


def minor(matrix, row, col):
    n, m = len(matrix), len(matrix[0])
    new_a = []
    for i in range(n):
        for j in range(m):
            if i == row or j == col:
                continue
            if not new_a or len(new_a[-1]) == m - 1:
                new_a.append([])
            new_a[-1].append(matrix[i][j])
    new_mat = _matrix.Matrix(new_a, t=matrix.t)
    return new_mat.det()


def cof(matrix, row, col):
    return matrix.sign_cof(row, col) * matrix.minor(row, col)


def minorMat(matrix):
    new = [[0] * len(matrix[0]) for _ in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            new[i][j] = matrix.minor(i, j)
    return _matrix.Matrix(new, t=matrix.t)


def cofMat(matrix):
    new = [[0] * len(matrix[0]) for _ in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            new[i][j] = matrix.cof(i, j)
    return _matrix.Matrix(new, t=matrix.t)


def adj(matrix):
    return matrix.cofMat().T()


def det(matrix):
    if len(matrix) == 0:
        return 1
    n, m = len(matrix), len(matrix[0])
    assert n == m
    if n == 2:
        return matrix.det2d()
    if n == 3:
        return matrix.det3d()
    return (matrix * matrix.adj())[0][0]


def inv_MIA(matrix):
    A = [row[:] for row in matrix]
    I_MAT = _matrix.Matrix.im(len(matrix), matrix.t)
    for i in range(len(matrix)):
        pivot = A[i][i]
        if pivot == 0:
            for j in range(i + 1, len(matrix)):
                if A[j][i] != 0:
                    A[i], A[j] = A[j], A[i]
                    I_MAT[i], I_MAT[j] = I_MAT[j], I_MAT[i]
                    pivot = A[i][i]
                    break
        if pivot == 0:
            raise ValueError("Matrix is singular and cannot be inverted.")
        for j in range(len(matrix)):
            A[i][j] /= pivot
            I_MAT[i][j] /= pivot
        for j in range(len(matrix)):
            if i != j:
                factor = A[j][i]
                for k in range(len(matrix)):
                    A[j][k] -= factor * A[i][k]
                    I_MAT[j][k] -= factor * I_MAT[i][k]
    return _matrix.Matrix(I_MAT, t=matrix.t)


def inv(matrix):
    if len(matrix) == 0:
        return _matrix.Matrix([], t=matrix.t)
    assert len(matrix) == len(matrix[0]), "Must be a square"
    if len(matrix) == 1:
        assert matrix[0][0] != 0, "No inverse"
        return _matrix.Matrix([[Fraction(1, matrix[0][0])]], t=matrix.t)
    if len(matrix) == 2:
        return matrix.inv2d()
    new_mat = matrix._copyMat()
    return new_mat.adj() * (1 / new_mat.det())


def inv2d(matrix):
    new_mat = _matrix.Matrix(
        [[matrix[1][1], -matrix[0][1]], [-matrix[1][0], matrix[0][0]]], t=matrix.t
    )
    return new_mat * (1 / (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]))


def isinv(a: "_matrix.Matrix", b: "_matrix.Matrix"):
    if not isinstance(a, _matrix.Matrix) or not isinstance(b, _matrix.Matrix):
        return False
    n = len(a)
    if not n or len(b) != n:
        return False
    if any(len(row) != n for matrix in (a, b) for row in matrix):
        return False
    identity = _matrix.Matrix.imat(n, a.t)
    return a * b == identity and b * a == identity


def solve(matrix, b):
    detA = matrix.det()
    new_a = [
        Fraction(matrix.changeCol(i, 0, b).det(), detA) for i in range(len(matrix))
    ]
    return _matrix.Matrix([new_a])


def solveSelf(matrix):
    A = _matrix.Matrix([row[:-1] for row in matrix], matrix.t)
    b = _matrix.Matrix([[row[-1]] for row in matrix], matrix.t)
    return A.solve(b)


def _rowNonZero(matrix, rows):
    for i in range(len(rows)):
        if rows[i] != 0:
            return i
    return len(rows)


def _rearrange(matrix, arr):
    arr.sort(key=lambda r: matrix._rowNonZero(r))


def isrref(matrix, arr=None):
    if arr is None:
        arr = matrix.a
    if not arr:
        return True
    if any(len(row) != len(arr[0]) for row in arr):
        return False

    previous_pivot = -1
    found_zero_row = False
    for row_index, row in enumerate(arr):
        pivot = matrix._rowNonZero(row)
        if pivot == len(row):
            found_zero_row = True
            continue
        if found_zero_row or pivot <= previous_pivot or row[pivot] != 1:
            return False
        if any(other[pivot] != 0 for i, other in enumerate(arr) if i != row_index):
            return False
        previous_pivot = pivot
    return True


def rref(matrix, tol=1e-12):

    if tol < 0:
        raise ValueError("Tolerance must be nonnegative.")
    A = [row[:] for row in matrix]
    if not A:
        return _matrix.Matrix([], t=matrix.t)
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
            A[i] = [
                value - factor * pivot_value
                for value, pivot_value in zip(A[i], A[pivot_row])
            ]
            A[i][col] = 0
        pivot_row += 1

    return _matrix.Matrix(A, t=Fraction if exact else matrix.t)

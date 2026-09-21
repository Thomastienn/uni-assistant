from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, Any

from linear_algebra import config
from linear_algebra._matrix_types import Row, Rows, ScalarParser

if TYPE_CHECKING:
    from linear_algebra.matrix import Matrix


def im(n: int, t: ScalarParser = eval) -> Rows:
    return [[t("1") if i == j else t("0") for j in range(n)] for i in range(n)]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    assert len(a[0]) == len(b), "not matching size"
    result = [[a.t("0")] * len(b[0]) for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                result[i][j] += a[i][k] * b[k][j]
    return a._new(result)


def scale(rows: Matrix, scalar: Any) -> Matrix:
    return rows._new([[value * scalar for value in row] for row in rows])


def add(a: Matrix, b: Matrix, delta: Any = 1) -> Matrix | None:
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        return None
    return a._new([
        [a[i][j] + b[i][j] * delta for j in range(len(a[0]))]
        for i in range(len(a))
    ])


def transpose(rows: Matrix) -> Matrix:
    return rows._new([list(row) for row in zip(*rows)])


def change_col(rows: Matrix, acol: int, bcol: int, other: Matrix) -> Matrix:
    result = [row[:] for row in rows]
    for i in range(len(rows)):
        result[i][acol] = other[i][bcol]
    return rows._new(result)


def concat(a: Matrix, b: Matrix) -> Matrix:
    assert len(a) == len(b), "not matching size"
    return a._new([row + other for row, other in zip(a, b)])


def det2d(matrix: Matrix) -> Any:
    n, m = len(matrix), len(matrix[0])
    assert n == 2 and m == 2

    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def det3d(matrix: Matrix) -> Any:
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


def sign_cof(row: int, col: int) -> int:
    return -1 if (row + col) & 1 else 1


def minor(matrix: Matrix, row: int, col: int) -> Any:
    n, m = len(matrix), len(matrix[0])
    new_a = []
    for i in range(n):
        for j in range(m):
            if i == row or j == col:
                continue
            if not new_a or len(new_a[-1]) == m - 1:
                new_a.append([])
            new_a[-1].append(matrix[i][j])
    return det(matrix._new(new_a))


def cof(matrix: Matrix, row: int, col: int) -> Any:
    return sign_cof(row, col) * minor(matrix, row, col)


def minorMat(matrix: Matrix) -> Matrix:
    new = [[0] * len(matrix[0]) for _ in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            new[i][j] = minor(matrix, i, j)
    return matrix._new(new)


def cofMat(matrix: Matrix) -> Matrix:
    new = [[0] * len(matrix[0]) for _ in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            new[i][j] = cof(matrix, i, j)
    return matrix._new(new)


def adj(matrix: Matrix) -> Matrix:
    return transpose(cofMat(matrix))


def det(matrix: Matrix) -> Any:
    if len(matrix) == 0:
        return 1
    n, m = len(matrix), len(matrix[0])
    assert n == m
    if n == 2:
        return det2d(matrix)
    if n == 3:
        return det3d(matrix)
    return matmul(matrix, adj(matrix))[0][0]


def inv_MIA(matrix: Matrix) -> Matrix:
    A: Rows = [row[:] for row in matrix]
    I_MAT = im(len(matrix), matrix.t)
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
    return matrix._new(I_MAT)


def inv(matrix: Matrix) -> Matrix:
    if len(matrix) == 0:
        return matrix._new([])
    assert len(matrix) == len(matrix[0]), "Must be a square"
    if len(matrix) == 1:
        assert matrix[0][0] != 0, "No inverse"
        return matrix._new([[Fraction(1, matrix[0][0])]])
    if len(matrix) == 2:
        return inv2d(matrix)
    return scale(adj(matrix), 1 / det(matrix))


def inv2d(matrix: Matrix) -> Matrix:
    new_mat = [[matrix[1][1], -matrix[0][1]], [-matrix[1][0], matrix[0][0]]]
    return scale(matrix._new(new_mat), 1 / det2d(matrix))


def isinv(a: Matrix, b: Matrix) -> bool:
    n = len(a)
    if not n or len(b) != n:
        return False
    if any(len(row) != n for matrix in (a, b) for row in matrix):
        return False
    identity = im(n, a.t)
    return matmul(a, b) == identity and matmul(b, a) == identity


def solve(matrix: Matrix, b: Matrix) -> Matrix:
    detA = det(matrix)
    new_a = [
        Fraction(det(change_col(matrix, i, 0, b)), detA) for i in range(len(matrix))
    ]
    return matrix._new([new_a])


def solveSelf(matrix: Matrix) -> Matrix:
    A = matrix._new([row[:-1] for row in matrix])
    b = matrix._new([[row[-1]] for row in matrix])
    return solve(A, b)


def _first_nonzero(row: Row) -> int:
    for i, value in enumerate(row):
        if value != 0:
            return i
    return len(row)


def isrref(arr: Rows) -> bool:
    if not arr:
        return True
    if any(len(row) != len(arr[0]) for row in arr):
        return False

    previous_pivot = -1
    found_zero_row = False
    for row_index, row in enumerate(arr):
        pivot = _first_nonzero(row)
        if pivot == len(row):
            found_zero_row = True
            continue
        if found_zero_row or pivot <= previous_pivot or row[pivot] != 1:
            return False
        if any(other[pivot] != 0 for i, other in enumerate(arr) if i != row_index):
            return False
        previous_pivot = pivot
    return True


def rref(matrix: Matrix) -> Matrix:
    tol = config.TOLERANCE
    if tol < 0:
        raise ValueError("Tolerance must be nonnegative.")
    A: Rows = [row[:] for row in matrix]
    if not A:
        return matrix._new([])
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

    return matrix._new(A, Fraction if exact else matrix.t)


def col_space(matrix: Matrix) -> list[Matrix]:
    reduced = rref(matrix)
    basis = []
    for row in reduced:
        pivot = _first_nonzero(row)
        if pivot < len(row):
            basis.append(matrix._new([[r[pivot]] for r in matrix]))
    return basis


def row_space(matrix: Matrix) -> list[Matrix]:
    reduced = rref(matrix)
    return [reduced._new([[value] for value in row]) for row in reduced if any(row)]


def null_space(matrix: Matrix) -> list[Matrix]:
    reduced = rref(matrix)
    cols = len(reduced[0]) if reduced else 0
    pivots = {_first_nonzero(row): row for row in reduced if any(row)}
    basis = []
    for free_col in range(cols):
        if free_col in pivots:
            continue
        vector = [[reduced.t("0")] for _ in range(cols)]
        vector[free_col][0] = reduced.t("1")
        for pivot, row in pivots.items():
            vector[pivot][0] = -row[free_col]
        basis.append(reduced._new(vector))
    return basis

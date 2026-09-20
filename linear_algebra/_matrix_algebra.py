from fractions import Fraction
from typing import Any

from linear_algebra._matrix_types import Row, Rows, ScalarParser


def im(n: int, t: ScalarParser = eval) -> Rows:
    return [[t("1") if i == j else t("0") for j in range(n)] for i in range(n)]


def matmul(a: Rows, b: Rows, t: ScalarParser = eval) -> Rows:
    assert len(a[0]) == len(b), "not matching size"
    result = [[t("0")] * len(b[0]) for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                result[i][j] += a[i][k] * b[k][j]
    return result


def scale(rows: Rows, scalar: Any) -> Rows:
    return [[value * scalar for value in row] for row in rows]


def add(a: Rows, b: Rows, delta: Any = 1) -> Rows | None:
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        return None
    return [[a[i][j] + b[i][j] * delta for j in range(len(a[0]))]
            for i in range(len(a))]


def transpose(rows: Rows) -> Rows:
    return [list(row) for row in zip(*rows)]


def change_col(rows: Rows, acol: int, bcol: int, other: Rows) -> Rows:
    result = [row[:] for row in rows]
    for i in range(len(rows)):
        result[i][acol] = other[i][bcol]
    return result


def concat(a: Rows, b: Rows) -> Rows:
    assert len(a) == len(b), "not matching size"
    return [row + other for row, other in zip(a, b)]


def det2d(matrix: Rows) -> Any:
    n, m = len(matrix), len(matrix[0])
    assert n == 2 and m == 2

    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def det3d(matrix: Rows, t: ScalarParser = eval) -> Any:
    assert len(matrix) == len(matrix[0]) == 3
    s = t("0")
    for off in range(3):
        pro1 = t("1")
        pro2 = t("1")
        for i in range(3):
            pro1 *= matrix[i][(i + off) % 3]
            pro2 *= matrix[i][(off - i) % 3]
        s += pro1 - pro2
    return s


def sign_cof(row: int, col: int) -> int:
    return -1 if (row + col) & 1 else 1


def minor(matrix: Rows, row: int, col: int, t: ScalarParser = eval) -> Any:
    n, m = len(matrix), len(matrix[0])
    new_a = []
    for i in range(n):
        for j in range(m):
            if i == row or j == col:
                continue
            if not new_a or len(new_a[-1]) == m - 1:
                new_a.append([])
            new_a[-1].append(matrix[i][j])
    return det(new_a, t)


def cof(matrix: Rows, row: int, col: int, t: ScalarParser = eval) -> Any:
    return sign_cof(row, col) * minor(matrix, row, col, t)


def minorMat(matrix: Rows, t: ScalarParser = eval) -> Rows:
    new = [[0] * len(matrix[0]) for _ in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            new[i][j] = minor(matrix, i, j, t)
    return new


def cofMat(matrix: Rows, t: ScalarParser = eval) -> Rows:
    new = [[0] * len(matrix[0]) for _ in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            new[i][j] = cof(matrix, i, j, t)
    return new


def adj(matrix: Rows, t: ScalarParser = eval) -> Rows:
    return transpose(cofMat(matrix, t))


def det(matrix: Rows, t: ScalarParser = eval) -> Any:
    if len(matrix) == 0:
        return 1
    n, m = len(matrix), len(matrix[0])
    assert n == m
    if n == 2:
        return det2d(matrix)
    if n == 3:
        return det3d(matrix, t)
    return matmul(matrix, adj(matrix, t), t)[0][0]


def inv_MIA(matrix: Rows, t: ScalarParser = eval) -> Rows:
    A: Rows = [row[:] for row in matrix]
    I_MAT = im(len(matrix), t)
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
    return I_MAT


def inv(matrix: Rows, t: ScalarParser = eval) -> Rows:
    if len(matrix) == 0:
        return []
    assert len(matrix) == len(matrix[0]), "Must be a square"
    if len(matrix) == 1:
        assert matrix[0][0] != 0, "No inverse"
        return [[Fraction(1, matrix[0][0])]]
    if len(matrix) == 2:
        return inv2d(matrix)
    return scale(adj(matrix, t), 1 / det(matrix, t))


def inv2d(matrix: Rows) -> Rows:
    new_mat = [[matrix[1][1], -matrix[0][1]], [-matrix[1][0], matrix[0][0]]]
    return scale(new_mat, 1 / det2d(matrix))


def isinv(a: Rows, b: Rows, t: ScalarParser = eval, other_t: ScalarParser = eval) -> bool:
    n = len(a)
    if not n or len(b) != n:
        return False
    if any(len(row) != n for matrix in (a, b) for row in matrix):
        return False
    identity = im(n, t)
    return matmul(a, b, t) == identity and matmul(b, a, other_t) == identity


def solve(matrix: Rows, b: Rows, t: ScalarParser = eval) -> Rows:
    detA = det(matrix, t)
    new_a = [
        Fraction(det(change_col(matrix, i, 0, b), t), detA) for i in range(len(matrix))
    ]
    return [new_a]


def solveSelf(matrix: Rows, t: ScalarParser = eval) -> Rows:
    A = [row[:-1] for row in matrix]
    b = [[row[-1]] for row in matrix]
    return solve(A, b, t)


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


def rref(matrix: Rows, t: ScalarParser = eval, tol: float = 1e-12) -> tuple[Rows, ScalarParser]:
    if tol < 0:
        raise ValueError("Tolerance must be nonnegative.")
    A: Rows = [row[:] for row in matrix]
    if not A:
        return [], t
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

    return A, Fraction if exact else t


def col_space(matrix: Rows, t: ScalarParser = eval, tol: float = 1e-12) -> list[Rows]:
    reduced, _ = rref(matrix, t, tol)
    basis = []
    for row in reduced:
        pivot = _first_nonzero(row)
        if pivot < len(row):
            basis.append([[r[pivot]] for r in matrix])
    return basis


def row_space(matrix: Rows, t: ScalarParser = eval, tol: float = 1e-12) -> tuple[list[Rows], ScalarParser]:
    reduced, result_type = rref(matrix, t, tol)
    return [[[value] for value in row] for row in reduced if any(row)], result_type

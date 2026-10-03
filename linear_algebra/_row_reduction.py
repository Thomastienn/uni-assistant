from __future__ import annotations

from typing import TYPE_CHECKING

from sympy import Expr, Float, sympify
from sympy.polys.constructor import construct_domain

from linear_algebra import config
from linear_algebra._matrix_algebra import divide, is_zero, require_square, simplify_scalar

if TYPE_CHECKING:
    from linear_algebra.matrix import Matrix


def reduce_rows(matrix: Matrix) -> tuple[Matrix, list[int]]:
    if config.TOLERANCE < 0:
        raise ValueError("Tolerance must be nonnegative.")
    rows = matrix.copy().data
    if any(isinstance(value, Expr) and value.free_symbols for row in rows for value in row):
        raise ValueError("Row reduction requires numeric entries; substitute symbols first.")
    approximate = any(
        isinstance(value, (float, complex)) or isinstance(value, Expr) and value.has(Float)
        for row in rows for value in row
    )
    domain = None
    entries = [value for row in rows for value in row]
    if not approximate and any(isinstance(value, Expr) for value in entries):
        if all(sympify(value).is_algebraic is True for value in entries):
            # Polynomial arithmetic reduces powers of exact roots during elimination.
            domain, entries = construct_domain(entries, extension=True, field=True)
            rows = [entries[i * matrix.ncols:(i + 1) * matrix.ncols] for i in range(matrix.nrows)]
    zero, one = (domain.zero, domain.one) if domain is not None else (0, 1)

    def entry_is_zero(value):
        return value == zero if domain is not None else is_zero(value)

    pivots = []
    pivot_row = 0
    for col in range(matrix.ncols):
        candidates = [i for i in range(pivot_row, matrix.nrows) if not entry_is_zero(rows[i][col])]
        if not candidates:
            for i in range(pivot_row, matrix.nrows):
                rows[i][col] = zero
            continue

        # Partial pivoting helps floats; exact arithmetic only needs a nonzero pivot.
        chosen = max(candidates, key=lambda i: abs(complex(rows[i][col]))) if approximate else candidates[0]
        rows[pivot_row], rows[chosen] = rows[chosen], rows[pivot_row]
        pivot = rows[pivot_row][col]
        rows[pivot_row] = [divide(value, pivot) for value in rows[pivot_row]]
        rows[pivot_row][col] = one

        for i in range(matrix.nrows):
            if i == pivot_row:
                continue
            factor = rows[i][col]
            rows[i] = [
                simplify_scalar(value - factor * pivot_value)
                for value, pivot_value in zip(rows[i], rows[pivot_row])
            ]
            rows[i][col] = zero

        pivots.append(col)
        pivot_row += 1
        if pivot_row == matrix.nrows:
            break

    if domain is not None:
        rows = [[domain.to_sympy(value) for value in row] for row in rows]
    else:
        rows = [[0 if is_zero(value) else value for value in row] for row in rows]
    return matrix._new(rows), pivots


def null_space(matrix: Matrix) -> list[Matrix]:
    reduced, pivots = reduce_rows(matrix)
    basis = []
    for free_col in range(matrix.ncols):
        if free_col in pivots:
            continue
        vector = [0] * matrix.ncols
        vector[free_col] = 1
        for row, pivot_col in enumerate(pivots):
            vector[pivot_col] = -reduced[row][free_col]
        basis.append(matrix.vector(*vector))
    return basis


def col_space(matrix: Matrix) -> list[Matrix]:
    _, pivots = reduce_rows(matrix)
    return [matrix.vector(*(row[col] for row in matrix)) for col in pivots]


def row_space(matrix: Matrix) -> list[Matrix]:
    reduced, pivots = reduce_rows(matrix)
    return [matrix.vector(*reduced[i]) for i in range(len(pivots))]


def solve(matrix: Matrix, rhs: Matrix) -> Matrix:
    if rhs.nrows != matrix.nrows or rhs.ncols == 0:
        raise ValueError("Right-hand side must have matching rows and at least one column.")
    reduced, pivots = reduce_rows(matrix.augment(rhs))
    if any(col >= matrix.ncols for col in pivots):
        raise ValueError("System is inconsistent.")
    if len(pivots) != matrix.ncols:
        raise ValueError("System has infinitely many solutions.")
    return matrix._new([row[matrix.ncols:] for row in reduced][:matrix.ncols])


def inverse(matrix: Matrix) -> Matrix:
    require_square(matrix)
    if matrix.nrows == 0:
        return matrix.copy()
    return solve(matrix, matrix.identity(matrix.nrows))


def is_rref(matrix: Matrix) -> bool:
    previous_pivot = -1
    zero_row_seen = False
    for i, row in enumerate(matrix):
        pivot = next((j for j, value in enumerate(row) if not is_zero(value)), None)
        if pivot is None:
            zero_row_seen = True
            continue
        if zero_row_seen or pivot <= previous_pivot or not is_zero(row[pivot] - 1):
            return False
        if any(not is_zero(other[pivot]) for k, other in enumerate(matrix) if k != i):
            return False
        previous_pivot = pivot
    return True

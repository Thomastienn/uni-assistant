from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, Any

from sympy import Expr, Float, simplify

from linear_algebra import config
from linear_algebra.polynomial import reduce_root_expression

if TYPE_CHECKING:
    from linear_algebra.matrix import Matrix


def simplify_scalar(value: Any) -> Any:
    reduced = reduce_root_expression(value)
    if reduced is not None:
        return reduced
    return simplify(value) if isinstance(value, Expr) else value


def is_zero(value: Any, tolerance: float | None = None) -> bool:
    tolerance = config.TOLERANCE if tolerance is None else tolerance
    if tolerance < 0:
        raise ValueError("Tolerance must be nonnegative.")
    reduced = reduce_root_expression(value)
    if reduced is not None:
        return reduced == 0
    value = simplify_scalar(value)
    if isinstance(value, (float, complex)):
        return abs(value) <= tolerance
    if isinstance(value, Expr):
        if value.free_symbols:
            raise ValueError("Row reduction requires numeric entries; substitute symbols first.")
        if value.has(Float):
            return abs(complex(value)) <= tolerance
        if value.is_zero is not None:
            return bool(value.is_zero)
        result = value.equals(0)
        if result is None:
            raise ValueError(f"Cannot determine whether {value} is zero.")
        return bool(result)
    return value == 0


def divide(numerator: Any, denominator: Any) -> Any:
    if isinstance(numerator, (int, Fraction)) and isinstance(denominator, (int, Fraction)):
        return Fraction(numerator) / Fraction(denominator)
    return simplify_scalar(numerator / denominator)


def require_square(matrix: Matrix) -> None:
    if matrix.nrows != matrix.ncols:
        raise ValueError("Matrix must be square.")


def add(a: Matrix, b: Matrix) -> Matrix:
    if a.shape != b.shape:
        raise ValueError("Matrix shapes must match.")
    return a._new([[left + right for left, right in zip(r1, r2)] for r1, r2 in zip(a, b)])


def multiply(a: Matrix, b: Matrix) -> Matrix:
    if a.ncols != b.nrows:
        raise ValueError("Left column count must match right row count.")
    rows = [[0 for _ in range(b.ncols)] for _ in range(a.nrows)]
    for i in range(a.nrows):
        for j in range(b.ncols):
            for k in range(a.ncols):
                rows[i][j] += a[i][k] * b[k][j]
    return a._new(rows)


def determinant(matrix: Matrix) -> Any:
    require_square(matrix)
    n = matrix.nrows
    if n == 0:
        return 1
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    # ponytail: cofactor expansion is factorial; use elimination for large matrices.
    return sum(matrix[0][j] * cofactor(matrix, 0, j) for j in range(n))


def minor(matrix: Matrix, row: int, col: int) -> Any:
    require_square(matrix)
    if not 0 <= row < matrix.nrows or not 0 <= col < matrix.ncols:
        raise IndexError("Minor index is outside the matrix.")
    smaller = matrix._new([
        [value for j, value in enumerate(values) if j != col]
        for i, values in enumerate(matrix) if i != row
    ])
    return determinant(smaller)


def cofactor(matrix: Matrix, row: int, col: int) -> Any:
    return (-1) ** (row + col) * minor(matrix, row, col)


def cofactor_matrix(matrix: Matrix) -> Matrix:
    require_square(matrix)
    return matrix._new([
        [cofactor(matrix, i, j) for j in range(matrix.ncols)]
        for i in range(matrix.nrows)
    ])

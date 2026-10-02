from __future__ import annotations

from collections.abc import Callable, Iterator
from fractions import Fraction
from math import isfinite
from numbers import Number
from typing import Any

from sympy import Expr

from linear_algebra import _matrix_algebra as algebra
from linear_algebra import _matrix_spectral as spectral
from linear_algebra import _matrix_vectors as vectors
from linear_algebra import _row_reduction as reduction
from linear_algebra._matrix_types import Row, Rows
from linear_algebra.polynomial import Poly


class Matrix:
    def __init__(self, rows: Rows | Matrix) -> None:
        self.data = [list(row) for row in rows]
        if self.data and not self.data[0]:
            raise ValueError("Use Matrix([]) for the empty matrix; nonempty rows need columns.")
        if any(len(row) != self.ncols for row in self):
            raise ValueError("Matrix rows must have the same length.")
        for row in self:
            for value in row:
                if not isinstance(value, (Number, Expr)):
                    raise TypeError("Entries must be numbers or scalar symbolic expressions.")
                if isinstance(value, (float, complex)) and not isfinite(abs(value)):
                    raise ValueError("Entries must be finite.")
                if isinstance(value, Expr) and value.is_number and value.is_finite is not True:
                    raise ValueError("Entries must be finite.")

    @property
    def nrows(self) -> int:
        return len(self.data)

    @property
    def ncols(self) -> int:
        return len(self.data[0]) if self.data else 0

    @property
    def shape(self) -> tuple[int, int]:
        return self.nrows, self.ncols

    def __iter__(self) -> Iterator[Row]:
        return iter(self.data)

    def __len__(self) -> int:
        return self.nrows

    def __getitem__(self, index: int) -> Row:
        return self.data[index]

    def __repr__(self) -> str:
        return f"Matrix({self.data!r})"

    def __str__(self) -> str:
        return "\n".join(" ".join(map(str, row)) for row in self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return NotImplemented
        return self.data == other.data

    def is_close(self, other: Matrix, tolerance: float | None = None) -> bool:
        return self.shape == other.shape and all(
            algebra.is_zero(a - b, tolerance)
            for left, right in zip(self, other) for a, b in zip(left, right)
        )

    def _new(self, rows: Rows) -> Matrix:
        return Matrix(rows)

    def copy(self) -> Matrix:
        return Matrix(self)

    def astype(self, convert: Callable[[Any], Any]) -> Matrix:
        return self._new([[convert(value) for value in row] for row in self])

    @staticmethod
    def from_input(parse: Callable[[str], Any] = Fraction) -> Matrix:
        count = int(input())
        if count < 0:
            raise ValueError("Row count must be nonnegative.")
        return Matrix([[parse(value) for value in input().split()] for _ in range(count)])

    @staticmethod
    def zeros(rows: int, cols: int) -> Matrix:
        if rows < 0 or cols < 0 or (rows == 0) != (cols == 0):
            raise ValueError("Dimensions must be positive, or both zero for the empty matrix.")
        return Matrix([[0] * cols for _ in range(rows)])

    @staticmethod
    def identity(size: int) -> Matrix:
        result = Matrix.zeros(size, size)
        for i in range(size):
            result[i][i] = 1
        return result

    @staticmethod
    def vector(*values: Any) -> Matrix:
        return Matrix([[value] for value in values])

    @staticmethod
    def from_columns(columns: list[Matrix]) -> Matrix:
        if not columns:
            return Matrix([])
        for column in columns:
            vectors.require_vectors(columns[0], column)
        return Matrix([[column[i][0] for column in columns] for i in range(len(columns[0]))])

    def __add__(self, other: Matrix) -> Matrix:
        if not isinstance(other, Matrix):
            return NotImplemented
        return algebra.add(self, other)

    def __neg__(self) -> Matrix:
        return self * -1

    def __sub__(self, other: Matrix) -> Matrix:
        if not isinstance(other, Matrix):
            return NotImplemented
        return self + (-other)

    def __matmul__(self, other: Matrix) -> Matrix:
        if not isinstance(other, Matrix):
            return NotImplemented
        return algebra.multiply(self, other)

    def __mul__(self, scalar: Any) -> Matrix:
        if not isinstance(scalar, (Number, Expr)):
            return NotImplemented
        return self._new([[value * scalar for value in row] for row in self])

    def __rmul__(self, scalar: Any) -> Matrix:
        return self * scalar

    def __truediv__(self, scalar: Any) -> Matrix:
        if not isinstance(scalar, (Number, Expr)):
            return NotImplemented
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        return self._new([[algebra.divide(value, scalar) for value in row] for row in self])

    def __pow__(self, exponent: int) -> Matrix:
        if not isinstance(exponent, int):
            raise TypeError("Matrix exponent must be an integer.")
        algebra.require_square(self)
        if exponent < 0:
            return self.inverse() ** (-exponent)
        result = self.identity(self.nrows)
        base = self.copy()
        while exponent:
            if exponent % 2:
                result = result @ base
            base = base @ base
            exponent //= 2
        return result

    def transpose(self) -> Matrix:
        return self._new([list(row) for row in zip(*self)])

    def rotate90(self) -> Matrix:
        return self._new([row[::-1] for row in self.transpose()])

    def augment(self, other: Matrix) -> Matrix:
        if self.nrows != other.nrows:
            raise ValueError("Augmented matrices must have matching row counts.")
        return self._new([left + right for left, right in zip(self, other)])

    def __or__(self, other: Matrix) -> Matrix:
        if not isinstance(other, Matrix):
            return NotImplemented
        return self.augment(other)

    def swap_rows(self, first: int, second: int) -> Matrix:
        result = self.copy()
        result.data[first], result.data[second] = result.data[second], result.data[first]
        return result

    def swap_columns(self, first: int, second: int) -> Matrix:
        result = self.copy()
        for row in result:
            row[first], row[second] = row[second], row[first]
        return result

    def combine_rows(self, target: int, source: int, target_scale: Any = 1, source_scale: Any = 1) -> Matrix:
        result = self.copy()
        result.data[target] = [
            target_scale * a + source_scale * b for a, b in zip(self[target], self[source])
        ]
        return result

    def without_row(self, index: int) -> Matrix:
        result = self.copy()
        result.data.pop(index)
        return result

    def without_column(self, index: int) -> Matrix:
        result = self.copy()
        for row in result:
            row.pop(index)
        return self._new(result.data)

    def det(self) -> Any:
        return algebra.determinant(self)

    def minor(self, row: int, col: int) -> Any:
        return algebra.minor(self, row, col)

    def cofactor(self, row: int, col: int) -> Any:
        return algebra.cofactor(self, row, col)

    def cofactor_matrix(self) -> Matrix:
        return algebra.cofactor_matrix(self)

    def adjugate(self) -> Matrix:
        return self.cofactor_matrix().transpose()

    def inverse(self) -> Matrix:
        return reduction.inverse(self)

    def is_inverse_of(self, other: Matrix) -> bool:
        if self.shape != other.shape or self.nrows != self.ncols:
            return False
        identity = self.identity(self.nrows)
        return (self @ other).is_close(identity) and (other @ self).is_close(identity)

    def rref(self) -> Matrix:
        reduced, _ = reduction.reduce_rows(self)
        return reduced

    def is_rref(self) -> bool:
        return reduction.is_rref(self)

    def rank(self) -> int:
        _, pivots = reduction.reduce_rows(self)
        return len(pivots)

    def solve(self, rhs: Matrix) -> Matrix:
        return reduction.solve(self, rhs)

    def solve_augmented(self) -> Matrix:
        if self.ncols == 0:
            raise ValueError("Augmented system needs a right-hand-side column.")
        coefficients = self._new([row[:-1] for row in self])
        rhs = self.vector(*(row[-1] for row in self))
        return coefficients.solve(rhs)

    def null_space(self) -> list[Matrix]:
        return reduction.null_space(self)

    def column_space(self) -> list[Matrix]:
        return reduction.column_space(self)

    def row_space(self) -> list[Matrix]:
        return reduction.row_space(self)

    def is_vector(self) -> bool:
        return self.ncols == 1

    def dot(self, other: Matrix) -> Any:
        return vectors.dot(self, other)

    def cross(self, other: Matrix) -> Matrix:
        return vectors.cross(self, other)

    @staticmethod
    def is_orthogonal(basis: list[Matrix]) -> bool:
        return vectors.is_orthogonal(basis)

    def coordinates(self, basis: list[Matrix]) -> Matrix:
        return vectors.coordinates(self, basis)

    def orthogonal_coordinates(self, basis: list[Matrix]) -> Matrix:
        return vectors.orthogonal_coordinates(self, basis)

    def project(self, onto: Matrix) -> Matrix:
        return vectors.project(self, onto)

    def in_span(self, basis: list[Matrix]) -> bool:
        return vectors.in_span(self, basis)

    def charpoly(self) -> Poly:
        return spectral.charpoly(self)

    def eigenvalues(self) -> list[Any]:
        return spectral.eigenvalues(self)

    def eigenspace(self, eigenvalue: Any) -> list[Matrix]:
        return spectral.eigenspace(self, eigenvalue)

    def algebraic_multiplicity(self, eigenvalue: Any) -> int:
        return spectral.algebraic_multiplicity(self, eigenvalue)

    def geometric_multiplicity(self, eigenvalue: Any) -> int:
        return spectral.geometric_multiplicity(self, eigenvalue)

    def is_diagonalizable(self) -> bool:
        return spectral.is_diagonalizable(self)

    def diagonalize(self) -> tuple[Matrix, Matrix]:
        return spectral.diagonalize(self)

    def is_similar(self, other: Matrix) -> bool:
        return spectral.is_similar(self, other)

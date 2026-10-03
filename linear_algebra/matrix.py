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
        """Create a matrix from rectangular rows, copying each row.

        Reject nonnumeric entries, nonfinite numbers, and unequal row lengths.
        Use Matrix([]) for a 0 by 0 matrix; other empty shapes are unsupported.
        """
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
        """Return the number of rows."""
        return len(self.data)

    @property
    def ncols(self) -> int:
        """Return the number of columns."""
        return len(self.data[0]) if self.data else 0

    @property
    def shape(self) -> tuple[int, int]:
        """Return (number of rows, number of columns)."""
        return self.nrows, self.ncols

    def __iter__(self) -> Iterator[Row]:
        """Iterate over the underlying rows; editing a row changes this matrix."""
        return iter(self.data)

    def __len__(self) -> int:
        """Return the number of rows."""
        return self.nrows

    def __getitem__(self, index: int) -> Row:
        """Return the underlying row at index; editing it changes this matrix."""
        return self.data[index]

    def __repr__(self) -> str:
        """Return a representation showing the constructor and row data."""
        return f"Matrix({self.data!r})"

    def __str__(self) -> str:
        """Format rows on separate lines with space-separated entries."""
        return "\n".join(" ".join(map(str, row)) for row in self)

    def __eq__(self, other: object) -> bool:
        """Compare shapes and entries exactly, without a numerical tolerance."""
        if not isinstance(other, Matrix):
            return NotImplemented
        return self.data == other.data

    def is_close(self, other: Matrix, tolerance: float | None = None) -> bool:
        """Check equal shapes and entry differences against zero.

        Floating differences use the absolute tolerance (config.TOLERANCE by
        default); exact expressions are simplified and tested exactly.
        """
        return self.shape == other.shape and all(
            algebra.is_zero(a - b, tolerance)
            for left, right in zip(self, other) for a, b in zip(left, right)
        )

    def _new(self, rows: Rows) -> Matrix:
        """Construct a new Matrix from rows using the usual input validation."""
        return Matrix(rows)

    def copy(self) -> Matrix:
        """Return a matrix with independent row lists."""
        return Matrix(self)

    def astype(self, convert: Callable[[Any], Any]) -> Matrix:
        """Return a new matrix with convert applied to every entry."""
        return self._new([[convert(value) for value in row] for row in self])

    @staticmethod
    def from_input(parse: Callable[[str], Any] = Fraction) -> Matrix:
        """Read a row count, then one space-separated row per line.

        Parse each entry with parse; the default Fraction preserves values
        such as 1/3 exactly.
        """
        count = int(input())
        if count < 0:
            raise ValueError("Row count must be nonnegative.")
        return Matrix([[parse(value) for value in input().split()] for _ in range(count)])

    @staticmethod
    def zeros(rows: int, cols: int) -> Matrix:
        """Create a rows by cols zero matrix; dimensions must be positive or both zero."""
        if rows < 0 or cols < 0 or (rows == 0) != (cols == 0):
            raise ValueError("Dimensions must be positive, or both zero for the empty matrix.")
        return Matrix([[0] * cols for _ in range(rows)])

    @staticmethod
    def identity(size: int) -> Matrix:
        """Create a size by size identity matrix with ones on its diagonal."""
        result = Matrix.zeros(size, size)
        for i in range(size):
            result[i][i] = 1
        return result

    @staticmethod
    def vector(*values: Any) -> Matrix:
        """Create a column vector, e.g. Matrix.vector(3, 1) gives [[3], [1]].

        No values produces the 0 by 0 empty matrix.
        """
        return Matrix([[value] for value in values])

    @staticmethod
    def from_columns(columns: list[Matrix]) -> Matrix:
        """Build a matrix from equal-sized column vectors in the supplied order.

        An empty list produces Matrix([]). Invalid vector shapes raise ValueError.
        """
        if not columns:
            return Matrix([])
        for column in columns:
            vectors.require_vectors(columns[0], column)
        return Matrix([[column[i][0] for column in columns] for i in range(len(columns[0]))])

    def __add__(self, other: Matrix) -> Matrix:
        """Add matching matrix entries; shapes must match."""
        if not isinstance(other, Matrix):
            return NotImplemented
        return algebra.add(self, other)

    def __neg__(self) -> Matrix:
        """Return a new matrix with every entry negated."""
        return self * -1

    def __sub__(self, other: Matrix) -> Matrix:
        """Subtract matching matrix entries; shapes must match."""
        if not isinstance(other, Matrix):
            return NotImplemented
        return self + (-other)

    def __matmul__(self, other: Matrix) -> Matrix:
        """Multiply matrices using @; left columns must match right rows."""
        if not isinstance(other, Matrix):
            return NotImplemented
        return algebra.multiply(self, other)

    def __mul__(self, scalar: Any) -> Matrix:
        """Multiply every entry by a scalar; use @ for matrix multiplication."""
        if not isinstance(scalar, (Number, Expr)):
            return NotImplemented
        return self._new([[value * scalar for value in row] for row in self])

    def __rmul__(self, scalar: Any) -> Matrix:
        """Multiply every entry by a scalar written on the left."""
        return self * scalar

    def __truediv__(self, scalar: Any) -> Matrix:
        """Divide entries by a nonzero scalar, preserving exact integer fractions."""
        if not isinstance(scalar, (Number, Expr)):
            return NotImplemented
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        return self._new([[algebra.divide(value, scalar) for value in row] for row in self])

    def __pow__(self, exponent: int) -> Matrix:
        """Raise a square matrix to an integer power.

        Power zero gives the identity; negative powers require an inverse.
        """
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
        """Return a new matrix with rows and columns exchanged; do not conjugate entries."""
        return self._new([list(row) for row in zip(*self)])

    def rotate90(self) -> Matrix:
        """Return a new matrix rotated 90 degrees clockwise."""
        return self._new([row[::-1] for row in self.transpose()])

    def augment(self, other: Matrix) -> Matrix:
        """Join other to the right of this matrix; row counts must match."""
        if self.nrows != other.nrows:
            raise ValueError("Augmented matrices must have matching row counts.")
        return self._new([left + right for left, right in zip(self, other)])

    def __or__(self, other: Matrix) -> Matrix:
        """Form an augmented matrix [self | other], equivalent to augment(other)."""
        if not isinstance(other, Matrix):
            return NotImplemented
        return self.augment(other)

    def swap_rows(self, first: int, second: int) -> Matrix:
        """Return a copy with the two indexed rows exchanged."""
        result = self.copy()
        result.data[first], result.data[second] = result.data[second], result.data[first]
        return result

    def swap_columns(self, first: int, second: int) -> Matrix:
        """Return a copy with the two indexed columns exchanged."""
        result = self.copy()
        for row in result:
            row[first], row[second] = row[second], row[first]
        return result

    def combine_rows(self, target: int, source: int, target_scale: Any = 1, source_scale: Any = 1) -> Matrix:
        """Return a copy with target replaced by a linear combination of two rows.

        The new row is target_scale * old_target + source_scale * old_source.
        For R1 <- R1 - 2*R0, use combine_rows(1, 0, source_scale=-2).
        """
        result = self.copy()
        result.data[target] = [
            target_scale * a + source_scale * b for a, b in zip(self[target], self[source])
        ]
        return result

    def without_row(self, index: int) -> Matrix:
        """Return a copy with the indexed row removed."""
        result = self.copy()
        result.data.pop(index)
        return result

    def without_column(self, index: int) -> Matrix:
        """Return a copy with the indexed column removed.

        Removing the only column of a nonempty matrix raises ValueError because
        nonzero row counts with zero columns are unsupported.
        """
        result = self.copy()
        for row in result:
            row.pop(index)
        return self._new(result.data)

    def det(self) -> Any:
        """Return the determinant of a square matrix; the empty determinant is 1.

        Uses cofactor expansion, intended for small learning examples.
        """
        return algebra.determinant(self)

    def minor(self, row: int, col: int) -> Any:
        """Return the determinant after deleting row and col from a square matrix.

        Indices are zero-based and must be within the matrix.
        """
        return algebra.minor(self, row, col)

    def cofactor(self, row: int, col: int) -> Any:
        """Return (-1)**(row + col) times the minor at these zero-based indices."""
        return algebra.cofactor(self, row, col)

    def cofactor_matrix(self) -> Matrix:
        """Return the matrix of cofactors for a square matrix."""
        return algebra.cofactor_matrix(self)

    def adjugate(self) -> Matrix:
        """Return the transposed cofactor matrix of a square matrix."""
        return self.cofactor_matrix().transpose()

    def inverse(self) -> Matrix:
        """Return the inverse using row reduction.

        Raise ValueError if the matrix is not square or is singular.
        """
        return reduction.inverse(self)

    def is_inverse_of(self, other: Matrix) -> bool:
        """Check whether both matrix products equal the identity using is_close."""
        if self.shape != other.shape or self.nrows != self.ncols:
            return False
        identity = self.identity(self.nrows)
        return (self @ other).is_close(identity) and (other @ self).is_close(identity)

    def rref(self) -> Matrix:
        """Return reduced row echelon form using Gauss-Jordan elimination.

        Each pivot is 1 and is the only nonzero entry in its column. Integer
        and Fraction inputs stay exact; floating inputs use config.TOLERANCE.
        Substitute values for free symbols before calling.
        """
        reduced, _ = reduction.reduce_rows(self)
        return reduced

    def is_rref(self) -> bool:
        """Check whether this matrix is already in reduced row echelon form."""
        return reduction.is_rref(self)

    def rank(self) -> int:
        """Return the number of pivots, or independent columns, from row reduction."""
        _, pivots = reduction.reduce_rows(self)
        return len(pivots)

    def solve(self, rhs: Matrix) -> Matrix:
        """Return the unique X satisfying self @ X == rhs.

        The matrix may be rectangular. rhs must have the same number of rows
        and at least one column; each column is a separate right-hand side.
        Raise ValueError for inconsistent systems or infinitely many solutions.
        For a basis of solutions to self @ x == 0, use null_space().
        """
        return reduction.solve(self, rhs)

    def solve_augmented(self) -> Matrix:
        """Solve this augmented system, treating its last column as the right-hand side.

        Return a solution column; raise ValueError if no unique solution exists.
        """
        if self.ncols == 0:
            raise ValueError("Augmented system needs a right-hand-side column.")
        coefficients = self._new([row[:-1] for row in self])
        rhs = self.vector(*(row[-1] for row in self))
        return coefficients.solve(rhs)

    def null_space(self) -> list[Matrix]:
        """Return a basis of column vectors x satisfying self @ x == 0.

        All solutions are linear combinations of these vectors. Return [] when
        the zero vector is the only solution.
        """
        return reduction.null_space(self)

    def column_space(self) -> list[Matrix]:
        """Return independent original columns forming a basis of the column space."""
        return reduction.column_space(self)

    def row_space(self) -> list[Matrix]:
        """Return a row-space basis from nonzero RREF rows, each stored as a column vector."""
        return reduction.row_space(self)

    def is_vector(self) -> bool:
        """Return whether this matrix has exactly one column."""
        return self.ncols == 1

    def dot(self, other: Matrix) -> Any:
        """Return the sum of entry products for equal-sized column vectors.

        Complex entries are not conjugated; this is a bilinear dot product.
        Invalid vector shapes raise ValueError.
        """
        return vectors.dot(self, other)

    def cross(self, other: Matrix) -> Matrix:
        """Return the 3D cross product self x other as a column vector.

        Both inputs must be three-entry column vectors, otherwise raise ValueError.
        """
        return vectors.cross(self, other)

    @staticmethod
    def is_orthogonal(basis: list[Matrix]) -> bool:
        """Check whether every distinct pair of supplied vectors has dot product zero.

        Inputs must be equal-sized real column vectors. Zero vectors are allowed
        by this check; an empty list returns True. This does not check unit length
        or guarantee that the vectors form a basis.
        """
        return vectors.is_orthogonal(basis)

    def coordinates(self, basis: list[Matrix]) -> Matrix:
        """Find the coefficients that express this vector in the supplied basis.

        basis is an ordered list of independent column vectors of the same size
        as self. Return a coefficient column c satisfying
        Matrix.from_columns(basis) @ c == self.

        For basis = [Matrix.vector(1, 1), Matrix.vector(1, -1)],
        Matrix.vector(3, 1).coordinates(basis) returns the column [2, 1],
        because (3, 1) = 2*(1, 1) + 1*(1, -1).

        The basis need not span the whole ambient space, but self must lie in
        its span. Raise ValueError for incompatible shapes, dependent vectors,
        or a target outside the span. An empty basis accepts only the zero
        vector and returns Matrix([]).
        """
        return vectors.coordinates(self, basis)

    def orthogonal_coordinates(self, basis: list[Matrix]) -> Matrix:
        """Return the factors multiplying mutually orthogonal basis vectors.

        For each direction f, compute c = self.dot(f) / f.dot(f), returning the
        coefficients as a column in basis order. Inputs must be real column
        vectors of matching size; basis vectors must be nonzero and mutually
        orthogonal, but need not have length 1. Invalid inputs raise ValueError.

        If self lies in the span, these factors reconstruct self. Otherwise,
        they reconstruct its orthogonal projection onto that span.
        For self = Matrix.vector(3, 1) and
        basis = [Matrix.vector(1, 1), Matrix.vector(1, -1)], the factors are
        [2, 1]. An empty basis returns Matrix([]).
        """
        return vectors.orthogonal_coordinates(self, basis)

    def project(self, onto: Matrix) -> Matrix:
        """Return the projection vector along the nonzero direction onto.

        Compute onto * (self.dot(onto) / onto.dot(onto)). Both inputs must be
        real column vectors of the same size; invalid inputs raise ValueError.
        For example, Matrix.vector(3, 1).project(Matrix.vector(1, 1)) returns
        Matrix.vector(2, 2). To get just the scalar factor, use
        self.orthogonal_coordinates([onto])[0][0].
        """
        return vectors.project(self, onto)

    def in_span(self, basis: list[Matrix]) -> bool:
        """Check whether this vector is a linear combination of the supplied vectors.

        Accept dependent spanning sets, unlike coordinates(). Inputs must be
        matching column vectors. An empty list spans only the zero vector.
        """
        return vectors.in_span(self, basis)

    def charpoly(self) -> Poly:
        """Return the SymPy Poly det(x*I - self) for a square numeric matrix.

        Substitute values for free symbols before calling.
        """
        return spectral.charpoly(self)

    def eigenvalues(self) -> list[Any]:
        """Return all characteristic-polynomial roots, including repeated values.

        Requires a square matrix. Exact inputs may produce radicals, complex
        values, or CRootOf objects. Polynomial root finding targets algebraic
        numeric coefficients and may reject transcendental coefficients.
        """
        return spectral.eigenvalues(self)

    def eigenspace(self, eigenvalue: Any) -> list[Matrix]:
        """Return a column-vector basis of solutions to self @ v == eigenvalue * v.

        Requires a square matrix. Return [] if the value is not an eigenvalue.
        """
        return spectral.eigenspace(self, eigenvalue)

    def algebraic_multiplicity(self, eigenvalue: Any) -> int:
        """Count occurrences of eigenvalue as a root of the characteristic polynomial."""
        return spectral.algebraic_multiplicity(self, eigenvalue)

    def geometric_multiplicity(self, eigenvalue: Any) -> int:
        """Return the dimension of the eigenspace, or zero for a non-eigenvalue."""
        return spectral.geometric_multiplicity(self, eigenvalue)

    def is_diagonalizable(self) -> bool:
        """Check whether a square matrix has a full eigenvector basis over the complex numbers."""
        return spectral.is_diagonalizable(self)

    def diagonalize(self) -> tuple[Matrix, Matrix]:
        """Return (P, D) with eigenvectors in P and matching eigenvalues on diagonal D.

        They satisfy self @ P == P @ D, so self = P @ D @ P.inverse().
        Works over the complex numbers. Raise ValueError for a nonsquare matrix
        or when there are too few independent eigenvectors.
        """
        return spectral.diagonalize(self)

    def is_similar(self, other: Matrix) -> bool:
        """Check whether square matrices represent the same map in different bases.

        Return whether other = P.inverse() @ self @ P for some invertible P
        over the complex numbers. Compare characteristic polynomials and Jordan
        block sizes through ranks; equal eigenvalues alone are insufficient.
        Nonsquare inputs raise ValueError; different sizes return False.
        """
        return spectral.is_similar(self, other)

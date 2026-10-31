from __future__ import annotations

from itertools import combinations
from numbers import Real
from typing import Any

from sympy import Expr

from linear_algebra import config
from linear_algebra._matrix_algebra import divide, is_zero
from linear_algebra.matrix import Matrix
from linear_algebra.polynomial import Poly


def require_vectors(a: Matrix, b: Matrix) -> None:
    if not a.is_vector() or not b.is_vector() or a.shape != b.shape:
        raise ValueError("Expected column vectors of the same size.")


def dot(a: Matrix, b: Matrix) -> Any:
    require_vectors(a, b)
    return sum(left[0] * right[0] for left, right in zip(a, b))


def cross(a: Matrix, b: Matrix) -> Matrix:
    require_vectors(a, b)
    if a.nrows != 3:
        raise ValueError("Cross product requires 3D vectors.")
    x, y, z = (row[0] for row in a)
    u, v, w = (row[0] for row in b)
    return Matrix.vector(y * w - z * v, z * u - x * w, x * v - y * u)


def flatten(matrix: Matrix) -> Matrix:
    """Return matrix entries row by row as a column vector."""
    return Matrix.vector(*(value for row in matrix for value in row))


def vectorize(matrix: Matrix) -> Matrix:
    """Return matrix entries column by column as a column vector."""
    return flatten(matrix.T())


def require_real(vector: Matrix) -> None:
    if not vector.is_vector():
        raise ValueError("Expected a column vector.")
    for row in vector:
        value = row[0]
        if not isinstance(value, Real) and not (isinstance(value, Expr) and value.is_real is True):
            raise ValueError("Orthogonality and projection require real-valued vectors.")


def is_ortho(vectors: list[Matrix]) -> bool:
    if config.TOLERANCE < 0:
        raise ValueError("Tolerance must be nonnegative.")
    for vector in vectors:
        require_real(vector)
        require_vectors(vectors[0], vector)
    return all(is_zero(dot(a, b)) for a, b in combinations(vectors, 2))


def ortho_coords(vector: Matrix, basis: list[Matrix]) -> Matrix:
    require_real(vector)
    if not is_ortho(basis):
        raise ValueError("Basis vectors must be orthogonal.")
    coefficients = []
    for direction in basis:
        require_vectors(vector, direction)
        denominator = dot(direction, direction)
        if is_zero(denominator):
            raise ValueError("Basis vectors must be nonzero.")
        coefficients.append(divide(dot(vector, direction), denominator))
    return Matrix.vector(*coefficients)


def project(vector: Matrix, onto: Matrix) -> Matrix:
    coefficient = ortho_coords(vector, [onto])[0][0]
    return onto * coefficient


def as_vectors(target: Matrix, basis: list[Matrix]) -> tuple[Matrix, list[Matrix]]:
    """Return target and basis as column vectors, accepting same-shaped matrices."""
    if target.is_vector():
        for direction in basis:
            require_vectors(target, direction)
        return target, basis
    for direction in basis:
        if direction.shape != target.shape:
            raise ValueError("Expected matrices of the same shape.")
    return target.vectorize(), [direction.vectorize() for direction in basis]


def coords(target: Matrix | Poly, basis: list[Matrix] | list[Poly]) -> Matrix:
    """Return the coordinate column c satisfying target = sum(c_i * basis_i).

    Accept same-shaped Matrix objects, or univariate SymPy Poly objects with
    the same generator and numeric coefficients in characteristic zero.
    Polynomial coefficients are matched in increasing powers, padding missing
    terms with zeros, before solving with our handwritten matrix solver.

    The ordered basis must be independent and its span must contain target.
    Raise TypeError for unsupported or mixed types; raise ValueError for
    incompatible shapes or generators, unsupported coefficients or fields,
    dependent bases, or a target outside the span. An empty basis accepts only
    a zero target and returns Matrix([]).
    """
    if isinstance(target, Poly):
        polynomials = [target, *basis]
        if any(not isinstance(poly, Poly) for poly in polynomials):
            raise TypeError("A polynomial target requires a basis of Poly objects.")
        if any(len(poly.gens) != 1 or poly.gens != target.gens for poly in polynomials):
            raise ValueError("Polynomials must be univariate with the same generator.")
        if any(poly.domain.characteristic() != 0 for poly in polynomials):
            raise ValueError("Polynomial coordinates require characteristic-zero coefficients.")

        size = max(len(poly.all_coeffs()) for poly in polynomials)
        columns = [
            Matrix.vector(*(poly.nth(power) for power in range(size)))
            for poly in polynomials
        ]
        target, *basis = columns
    elif not isinstance(target, Matrix) or any(not isinstance(direction, Matrix) for direction in basis):
        raise TypeError("Target and basis must be all Matrix objects or all Poly objects.")

    if not basis:
        if all(is_zero(value) for row in target for value in row):
            return target._new([])
        raise ValueError("Target is outside the empty span.")
    target, directions = as_vectors(target, basis)
    return Matrix.from_cols(directions).solve(target)


def in_span(vector: Matrix, basis: list[Matrix]) -> bool:
    """Check membership in the span of matching vectors or matrices."""
    if not basis:
        return all(is_zero(value) for row in vector for value in row)
    target, directions = as_vectors(vector, basis)
    if target.shape == (0, 0):
        return True
    columns = Matrix.from_cols(directions)
    return columns.rank() == columns.augment(target).rank()

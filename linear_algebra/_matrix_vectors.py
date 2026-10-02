from __future__ import annotations

from itertools import combinations
from numbers import Real
from typing import TYPE_CHECKING, Any

from sympy import Expr

from linear_algebra import config
from linear_algebra._matrix_algebra import divide, is_zero

if TYPE_CHECKING:
    from linear_algebra.matrix import Matrix


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
    return a.vector(y * w - z * v, z * u - x * w, x * v - y * u)


def require_real(vector: Matrix) -> None:
    if not vector.is_vector():
        raise ValueError("Expected a column vector.")
    for row in vector:
        value = row[0]
        if not isinstance(value, Real) and not (isinstance(value, Expr) and value.is_real is True):
            raise ValueError("Orthogonality and projection require real-valued vectors.")


def is_orthogonal(vectors: list[Matrix]) -> bool:
    if config.TOLERANCE < 0:
        raise ValueError("Tolerance must be nonnegative.")
    for vector in vectors:
        require_real(vector)
        require_vectors(vectors[0], vector)
    return all(is_zero(dot(a, b)) for a, b in combinations(vectors, 2))


def orthogonal_coordinates(vector: Matrix, basis: list[Matrix]) -> Matrix:
    require_real(vector)
    if not is_orthogonal(basis):
        raise ValueError("Basis vectors must be orthogonal.")
    coefficients = []
    for direction in basis:
        require_vectors(vector, direction)
        denominator = dot(direction, direction)
        if is_zero(denominator):
            raise ValueError("Basis vectors must be nonzero.")
        coefficients.append(divide(dot(vector, direction), denominator))
    return vector.vector(*coefficients)


def project(vector: Matrix, onto: Matrix) -> Matrix:
    coefficient = orthogonal_coordinates(vector, [onto])[0][0]
    return onto * coefficient


def coordinates(vector: Matrix, basis: list[Matrix]) -> Matrix:
    if not vector.is_vector():
        raise ValueError("Target must be a column vector.")
    for direction in basis:
        require_vectors(vector, direction)
    if not basis:
        if all(is_zero(row[0]) for row in vector):
            return vector._new([])
        raise ValueError("Target is outside the empty span.")
    return vector.from_columns(basis).solve(vector)


def in_span(vector: Matrix, basis: list[Matrix]) -> bool:
    if not vector.is_vector():
        raise ValueError("Target must be a column vector.")
    for direction in basis:
        require_vectors(vector, direction)
    if not basis:
        return all(is_zero(row[0]) for row in vector)
    columns = vector.from_columns(basis)
    return columns.rank() == columns.augment(vector).rank()

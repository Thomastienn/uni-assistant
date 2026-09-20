from __future__ import annotations

from typing import TYPE_CHECKING, Any

from linear_algebra import _matrix_algebra as algebra

if TYPE_CHECKING:
    from linear_algebra.matrix import Matrix


def is_vector(matrix: Matrix) -> bool:
    return all(len(row) == 1 for row in matrix)


def vR(matrix: Matrix, pos: int) -> Any:
    assert is_vector(matrix), "not a vector, cannot use"
    return matrix[pos][0]


def dot(matrix: Matrix, other: Matrix) -> Any:
    if not is_vector(matrix) or not is_vector(other) or len(matrix) != len(other):
        raise ValueError("Dot product requires column vectors of the same size.")
    return sum(a[0] * b[0] for a, b in zip(matrix, other))


def cross(matrix: Matrix, other: Matrix) -> Matrix:
    if not is_vector(matrix) or not is_vector(other) or len(matrix) != 3 or len(other) != 3:
        raise ValueError("Cross product requires two 3D column vectors.")
    x, y, z = (row[0] for row in matrix)
    other_x, other_y, other_z = (row[0] for row in other)
    return matrix._new([
        [y * other_z - z * other_y],
        [z * other_x - x * other_z],
        [x * other_y - y * other_x],
    ])


def cB(matrix: Matrix, basis: list[Matrix]) -> Matrix:
    solve_mat = matrix._new([[] for _ in range(len(basis[0]))])
    for basis_vec in basis:
        solve_mat = algebra.concat(solve_mat, basis_vec)
    return algebra.transpose(algebra.solve(solve_mat, matrix))


def in_span(matrix: Matrix, basis: list[Matrix]) -> bool:
    if not is_vector(matrix):
        raise ValueError("Target must be a column vector.")
    for vector in basis:
        if not is_vector(vector) or len(vector) != len(matrix):
            raise ValueError("Basis vectors must be column vectors matching the target size.")
    if not basis or not matrix:
        return all(row[0] == 0 for row in matrix)
    augmented = basis[0]._copyMat()
    for vector in basis[1:]:
        augmented = algebra.concat(augmented, vector)
    augmented = algebra.concat(augmented, matrix)
    reduced = algebra.rref(augmented)
    for row in reduced:
        if all(value == 0 for value in row[:-1]) and row[-1] != 0:
            return False
    return True

from typing import Any

from linear_algebra import _matrix_algebra as algebra
from linear_algebra._matrix_types import Rows, ScalarParser


def is_vector(matrix: Rows) -> bool:
    return all(len(row) == 1 for row in matrix)


def vR(matrix: Rows, pos: int) -> Any:
    assert is_vector(matrix), "not a vector, cannot use"
    return matrix[pos][0]


def dot(matrix: Rows, other: Rows) -> Any:
    if not is_vector(matrix) or not is_vector(other) or len(matrix) != len(other):
        raise ValueError("Dot product requires column vectors of the same size.")
    return sum(a[0] * b[0] for a, b in zip(matrix, other))


def cross(matrix: Rows, other: Rows) -> Rows:
    if not is_vector(matrix) or not is_vector(other) or len(matrix) != 3 or len(other) != 3:
        raise ValueError("Cross product requires two 3D column vectors.")
    x, y, z = (row[0] for row in matrix)
    other_x, other_y, other_z = (row[0] for row in other)
    return [[y * other_z - z * other_y],
            [z * other_x - x * other_z],
            [x * other_y - y * other_x]]


def cB(matrix: Rows, basis: list[Rows], t: ScalarParser = eval) -> Rows:
    solve_mat: Rows = [[] for _ in range(len(basis[0]))]
    for basis_vec in basis:
        solve_mat = algebra.concat(solve_mat, basis_vec)
    return algebra.transpose(algebra.solve(solve_mat, matrix, t))


def in_span(matrix: Rows, basis: list[Rows]) -> bool:
    if not is_vector(matrix):
        raise ValueError("Target must be a column vector.")
    for vector in basis:
        if not is_vector(vector) or len(vector) != len(matrix):
            raise ValueError("Basis vectors must be column vectors matching the target size.")
    if not basis or not matrix:
        return all(row[0] == 0 for row in matrix)
    augmented = [row[:] for row in basis[0]]
    for vector in basis[1:]:
        augmented = algebra.concat(augmented, vector)
    augmented = algebra.concat(augmented, matrix)
    reduced, _ = algebra.rref(augmented)
    for row in reduced:
        if all(value == 0 for value in row[:-1]) and row[-1] != 0:
            return False
    return True

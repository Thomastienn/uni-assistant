from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from linear_algebra import _matrix_algebra as algebra
from linear_algebra._matrix_types import Row
from linear_algebra.polynomial import Poly

if TYPE_CHECKING:
    from linear_algebra.matrix import Matrix


def cA(matrix: Matrix) -> Poly:
    assert len(matrix) == len(matrix[0]), "need to be nxn"
    mat = algebra.add(algebra.scale(matrix.imat(len(matrix), matrix.t), Poly("x")), matrix, -1)
    assert mat is not None
    return algebra.det(mat)


def eigen_vals(matrix: Matrix) -> list[Any]:
    equal = cA(matrix)
    coefficients: Row = equal.coef[::-1]
    roots = np.roots(coefficients)
    return [round(float(x.real), 3) if abs(x.imag) < 1e-5 else x for x in roots]


def is_similar(matrix: Matrix, other: Matrix) -> bool:
    # Equal eigenvalues are necessary, but Jordan structure is not checked.
    n, m = len(matrix), len(matrix[0])
    k, l = len(other), len(other[0])
    if n != k or m != l:
        return False
    if algebra.det(matrix) != algebra.det(other):
        return False
    return sorted(eigen_vals(matrix)) == sorted(eigen_vals(other))


def eigen_vec(matrix: Matrix, eigen_val: Any) -> Matrix:
    # Returns the reduced augmented system, not an eigenvector basis.
    assert len(matrix) == len(matrix[0]), "need to be nxn"
    solve_mat = algebra.add(algebra.scale(matrix.imat(len(matrix), matrix.t), eigen_val), matrix, -1)
    assert solve_mat is not None
    augmented = algebra.concat(solve_mat, matrix.zero_vec(len(matrix), matrix.t))
    return algebra.rref(augmented)


def algebraic_multiplicity(matrix: Matrix, eigen_val: Any) -> int:
    return eigen_vals(matrix).count(eigen_val)


def geometric_multiplicity(matrix: Matrix, eigen_val: Any) -> int:
    reduced = eigen_vec(matrix, eigen_val)
    return sum(all(value == 0 for value in row) for row in reduced)


def is_diagnolizable(matrix: Matrix) -> bool:
    for eigen_val in eigen_vals(matrix):
        alg_mult = algebraic_multiplicity(matrix, eigen_val)
        geo_mult = geometric_multiplicity(matrix, eigen_val)
        if alg_mult != geo_mult:
            return False
    return True


def diag(matrix: Matrix) -> Matrix:
    assert is_diagnolizable(matrix), "not diagnolizable"
    eigenval = eigen_vals(matrix)
    result = matrix.imat(len(matrix), matrix.t)
    for i in range(len(matrix)):
        result[i][i] = eigenval[i]
    return result

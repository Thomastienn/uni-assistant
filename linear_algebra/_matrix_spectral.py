from typing import Any

import numpy as np
from linear_algebra import _matrix_algebra as algebra
from linear_algebra._matrix_types import Row, Rows, ScalarParser
from linear_algebra.polynomial import Poly


def cA(matrix: Rows, t: ScalarParser = eval) -> Poly:
    assert len(matrix) == len(matrix[0]), "need to be nxn"
    mat = algebra.add(algebra.scale(algebra.im(len(matrix), t), Poly("x")), matrix, -1)
    assert mat is not None
    return algebra.det(mat, t)


def eigen_vals(matrix: Rows, t: ScalarParser = eval) -> list[Any]:
    equal = cA(matrix, t)
    coefficients: Row = equal.coef[::-1]
    roots = np.roots(coefficients)
    return [round(float(x.real), 3) if abs(x.imag) < 1e-5 else x for x in roots]


def is_similar(matrix: Rows, other: Rows, t: ScalarParser = eval,
               other_t: ScalarParser = eval) -> bool:
    # Equal eigenvalues are necessary, but Jordan structure is not checked.
    n, m = len(matrix), len(matrix[0])
    k, l = len(other), len(other[0])
    if n != k or m != l:
        return False
    if algebra.det(matrix, t) != algebra.det(other, other_t):
        return False
    return sorted(eigen_vals(matrix, t)) == sorted(eigen_vals(other, other_t))


def eigen_vec(matrix: Rows, eigen_val: Any,
              t: ScalarParser = eval) -> tuple[Rows, ScalarParser]:
    # Returns the reduced augmented system, not an eigenvector basis.
    assert len(matrix) == len(matrix[0]), "need to be nxn"
    solve_mat = algebra.add(algebra.scale(algebra.im(len(matrix), t), eigen_val), matrix, -1)
    assert solve_mat is not None
    augmented = algebra.concat(solve_mat, [[t("0")] for _ in matrix])
    return algebra.rref(augmented, t)


def algebraic_multiplicity(matrix: Rows, eigen_val: Any, t: ScalarParser = eval) -> int:
    return eigen_vals(matrix, t).count(eigen_val)


def geometric_multiplicity(matrix: Rows, eigen_val: Any, t: ScalarParser = eval) -> int:
    reduced, _ = eigen_vec(matrix, eigen_val, t)
    return sum(all(value == 0 for value in row) for row in reduced)


def is_diagnolizable(matrix: Rows, t: ScalarParser = eval) -> bool:
    for eigen_val in eigen_vals(matrix, t):
        alg_mult = algebraic_multiplicity(matrix, eigen_val, t)
        geo_mult = geometric_multiplicity(matrix, eigen_val, t)
        if alg_mult != geo_mult:
            return False
    return True


def diag(matrix: Rows, t: ScalarParser = eval) -> Rows:
    assert is_diagnolizable(matrix, t), "not diagnolizable"
    eigenval = eigen_vals(matrix, t)
    result = algebra.im(len(matrix), t)
    for i in range(len(matrix)):
        result[i][i] = eigenval[i]
    return result

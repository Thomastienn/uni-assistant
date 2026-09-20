import numpy as np
from linear_algebra import matrix as _matrix
from linear_algebra.polynomial import Poly


def cA(matrix):
    if len(matrix) != len(matrix[0]):
        assert False, "need to be nxn"
    lambdaa = Poly("x")
    mat = (_matrix.Matrix.imat(len(matrix), matrix.t) * lambdaa) - matrix

    return mat.det()


def eigen_vals(matrix):
    equal = matrix.cA()
    roots = np.roots(equal.coef[::-1])
    ret = [round(float(x.real), 3) if abs(
        x.imag) < 1e-5 else x for x in roots]
    return ret


def is_similar(matrix, other):
    # Equal eigenvalues are necessary, but Jordan structure is not checked.
    n, m = len(matrix), len(matrix[0])
    k, l = len(other), len(other[0])
    if n != k or m != l:
        return False
    if matrix.det() != other.det():
        return False
    return sorted(matrix.eigen_vals()) == sorted(other.eigen_vals())


def eigen_vec(matrix, eigen_val):
    # Returns the reduced augmented system, not an eigenvector basis.
    if len(matrix) != len(matrix[0]):
        assert False, "need to be nxn"

    solve_mat = _matrix.Matrix.imat(len(matrix), matrix.t)
    solve_mat = solve_mat * eigen_val - matrix
    zero_vec = _matrix.Matrix.zero_vec(len(matrix), matrix.t)

    res_mat = solve_mat.concat(zero_vec)
    return res_mat.rref()


def algebraic_multiplicity(matrix, eigen_val):
    return matrix.eigen_vals().count(eigen_val)


def geometric_multiplicity(matrix, eigen_val):
    cnt = 0
    for row in matrix.eigen_vec(eigen_val):
        zeros = 0
        for c in row:
            if c == 0:
                zeros += 1
        cnt += zeros == len(row)

    return cnt


def is_diagnolizable(matrix):
    for eigen_val in matrix.eigen_vals():
        alg_mult = matrix.algebraic_multiplicity(eigen_val)
        geo_mult = matrix.geometric_multiplicity(eigen_val)
        if alg_mult != geo_mult:
            return False
    return True


def diag(matrix):
    if not matrix.is_diagnolizable():
        assert False, "not diagnolizable"
    eigenval = matrix.eigen_vals()
    new_a = _matrix.Matrix.imat(len(matrix), matrix.t)
    for i in range(len(matrix)):
        new_a[i][i] = eigenval[i]
    return _matrix.Matrix(new_a, t=matrix.t)

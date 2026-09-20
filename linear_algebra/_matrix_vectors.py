from linear_algebra import matrix as _matrix


def is_vector(matrix):
    for i in range(len(matrix)):
        if len(matrix[i]) != 1:
            return False
    return True


def vR(matrix, pos):
    if not matrix.is_vector():
        assert False, "not a vector, cannot use"
    return matrix[pos][0]


def dot(matrix, other: "_matrix.Matrix"):
    if not matrix.is_vector() or not other.is_vector() or len(matrix) != len(other):
        raise ValueError("Dot product requires column vectors of the same size.")
    return sum(a[0] * b[0] for a, b in zip(matrix, other))


def cross(matrix, other: "_matrix.Matrix"):
    if not matrix.is_vector() or not other.is_vector() or len(matrix) != 3 or len(other) != 3:
        raise ValueError("Cross product requires two 3D column vectors.")
    x, y, z = (row[0] for row in matrix)
    other_x, other_y, other_z = (row[0] for row in other)
    return _matrix.Matrix.mvec(
        y * other_z - z * other_y,
        z * other_x - x * other_z,
        x * other_y - y * other_x,
        t=matrix.t)


def cB(matrix, basis: list["_matrix.Matrix"]):
    solve_mat = _matrix.Matrix([[] * len(basis)
                       for _ in range(len(basis[0]))], t=matrix.t)
    for basis_vec in basis:
        solve_mat.concat(basis_vec, in_place=True)

    return solve_mat.solve(matrix).T()


def in_span(matrix, basis: list["_matrix.Matrix"]):
    if not matrix.is_vector():
        raise ValueError("Target must be a column vector.")
    for vector in basis:
        if not vector.is_vector() or len(vector) != len(matrix):
            raise ValueError("Basis vectors must be column vectors matching the target size.")

    if not basis or not matrix:
        return all(row[0] == 0 for row in matrix)

    augmented = basis[0]._copyMat()
    for vector in basis[1:]:
        augmented.concat(vector, in_place=True)
    augmented.concat(matrix, in_place=True)

    for row in augmented.rref():
        coefficients = row[:-1]
        target = row[-1]
        if all(value == 0 for value in coefficients) and target != 0:
            return False

    return True

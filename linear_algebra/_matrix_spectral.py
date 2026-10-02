from __future__ import annotations

from typing import TYPE_CHECKING, Any

from linear_algebra._matrix_algebra import is_zero, require_square
from linear_algebra.polynomial import Poly, x

if TYPE_CHECKING:
    from linear_algebra.matrix import Matrix


def charpoly(matrix: Matrix) -> Poly:
    require_square(matrix)
    if any(getattr(value, "free_symbols", set()) for row in matrix for value in row):
        raise ValueError("Spectral calculations require numeric entries; substitute symbols first.")
    shifted = matrix.identity(matrix.nrows) * x - matrix
    return Poly(shifted.det(), x, extension=True)


def eigenvalues(matrix: Matrix) -> list[Any]:
    # SymPy solves only the polynomial; all matrix algorithms remain ours.
    return charpoly(matrix).all_roots()


def eigenspace(matrix: Matrix, eigenvalue: Any) -> list[Matrix]:
    require_square(matrix)
    shifted = matrix - matrix.identity(matrix.nrows) * eigenvalue
    return shifted.null_space()


def algebraic_multiplicity(matrix: Matrix, eigenvalue: Any) -> int:
    return sum(is_zero(root - eigenvalue) for root in eigenvalues(matrix))


def geometric_multiplicity(matrix: Matrix, eigenvalue: Any) -> int:
    return len(eigenspace(matrix, eigenvalue))


def is_diagonalizable(matrix: Matrix) -> bool:
    roots = dict.fromkeys(eigenvalues(matrix))
    return sum(len(eigenspace(matrix, root)) for root in roots) == matrix.nrows


def diagonalize(matrix: Matrix) -> tuple[Matrix, Matrix]:
    vectors = []
    values = []
    for root in dict.fromkeys(eigenvalues(matrix)):
        basis = eigenspace(matrix, root)
        vectors.extend(basis)
        values.extend([root] * len(basis))
    if len(vectors) != matrix.nrows:
        raise ValueError("Matrix does not have a full eigenvector basis.")
    p = matrix.from_columns(vectors)
    d = matrix._new([
        [value if i == j else 0 for j in range(matrix.nrows)]
        for i, value in enumerate(values)
    ])
    return p, d


def is_similar(matrix: Matrix, other: Matrix) -> bool:
    require_square(matrix)
    require_square(other)
    if matrix.shape != other.shape:
        return False
    if not (charpoly(matrix) - charpoly(other)).is_zero:
        return False
    # Nullities of successive powers determine the Jordan block sizes.
    roots = eigenvalues(matrix)
    for root in dict.fromkeys(roots):
        left = matrix - matrix.identity(matrix.nrows) * root
        right = other - other.identity(other.nrows) * root
        left_power, right_power = left, right
        for _ in range(roots.count(root)):
            if left_power.rank() != right_power.rank():
                return False
            left_power = left_power @ left
            right_power = right_power @ right
    return True

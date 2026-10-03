from fractions import Fraction

from linear_algebra.matrix import Matrix
from linear_algebra.polynomial import Poly, x


def demo():
    diagonal = Matrix.diag(2, 3, 4)
    assert diagonal @ Matrix.vector(1, 2, 3) == Matrix.vector(2, 6, 12)

    polynomial = Poly(x**3 - 4*x**2 + 5*x - 2, x)
    assert polynomial.all_roots() == [1, 1, 2]

    a = Matrix([[2, 1], [1, 2]])
    assert a.charpoly() == Poly(x**2 - 4*x + 3, x)
    assert a.eigenvalues() == [1, 3]

    # Eigenvectors solve (A - lambda I)v = 0.
    for eigenvalue in a.eigenvalues():
        shifted = a - Matrix.identity(2) * eigenvalue
        for vector in a.eigenspace(eigenvalue):
            assert (shifted @ vector).is_close(Matrix.zeros(2, 1))

    # Eigenvectors become columns of P, in the same order as entries of D.
    p, d = a.diagonalize()
    assert (a @ p).is_close(p @ d)
    assert (p @ d @ p.inv()).is_close(a)

    rhs = Matrix.vector(1, 0)
    assert a.solve(rhs) == Matrix.vector(Fraction(2, 3), Fraction(-1, 3))
    assert a.is_inverse_of(a.inv())

    repeated = Matrix([[2, 1], [0, 2]])
    assert repeated.alg_mult(2) == 2
    assert repeated.geom_mult(2) == 1
    assert not repeated.can_diag()
    assert not repeated.is_similar(Matrix.identity(2) * 2)

    # This cubic has exact roots represented by CRootOf rather than simple radicals.
    cubic = Matrix([[0, 0, 1], [1, 0, 1], [0, 1, 0]])
    root = cubic.eigenvalues()[1]
    vector = cubic.eigenspace(root)[0]
    assert (cubic @ vector).is_close(vector * root)

    print("Characteristic polynomial:", a.charpoly().as_expr())
    print("Eigenvalues:", a.eigenvalues())
    print("Eigenbasis P:\n", p, sep="")
    print("Diagonal D:\n", d, sep="")
    print("All checks passed.")


if __name__ == "__main__":
    demo()

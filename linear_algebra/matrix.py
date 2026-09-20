from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal, overload

from linear_algebra import _matrix_algebra as _algebra
from linear_algebra import _matrix_spectral as _spectral
from linear_algebra import _matrix_vectors as _vectors
from linear_algebra._matrix_types import Row, Rows, ScalarParser
from linear_algebra.polynomial import Poly


class Matrix:
    def __init__(self, a: Rows | Matrix | None = None, t: ScalarParser = eval) -> None:
        self.t: ScalarParser = t
        self.a: Rows
        if a is None:
            self._get()
        else:
            self.a = a.a if isinstance(a, Matrix) else a

    def __iter__(self) -> Iterator[Row]:
        return iter(self.a)

    def __len__(self) -> int:
        return len(self.a)

    @overload
    def __getitem__(self, index: int) -> Row: ...

    @overload
    def __getitem__(self, index: slice) -> Rows: ...

    def __getitem__(self, index: int | slice) -> Row | Rows:
        return self.a[index]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (Matrix, list, tuple)):
            return NotImplemented
        n, m = len(self), len(self[0])
        k, l = len(other), len(other[0])
        if n != k or m != l:
            return False
        for i in range(n):
            for j in range(m):
                if self[i][j] != other[i][j]:
                    return False
        return True

    def __mul__(self, bmat: Any) -> Matrix:
        if isinstance(bmat, Matrix):
            rows = _algebra.matmul(self.a, bmat.a, self.t)
        else:
            rows = _algebra.scale(self.a, bmat)
        return Matrix(rows, self.t)

    def __rmul__(self, scalar: Any) -> Matrix:
        return self * scalar

    def __add__(self, bmat: Matrix) -> Matrix | None:
        return self.add(bmat, 1)

    def __sub__(self, bmat: Matrix) -> Matrix | None:
        return self.add(bmat, -1)

    def __pow__(self, n: int) -> Matrix:
        amat = self._copyMat()
        iden = Matrix.imat(len(self), self.t)
        while n:
            if n & 1:
                iden *= amat
            amat *= amat
            n >>= 1
        return iden

    def add(self, bmat: Matrix, delta: Any = 1) -> Matrix | None:
        rows = _algebra.add(self.a, bmat.a, delta)
        return None if rows is None else Matrix(rows, self.t)

    def _copyArr(self) -> Rows:
        return [row[:] for row in self]

    def _copyMat(self) -> Matrix:
        return Matrix(self._copyArr(), t=self.t)

    def removeRow(self, row: int) -> None:
        self.a.pop(row)

    def removeCol(self, col: int) -> None:
        n = len(self)
        for i in range(n):
            self[i].pop(col)

    def assignRow(self, bdelta: Any, brow: int, adelta: Any, arow: int, mainR: int) -> Matrix:
        new = self._copyMat()
        for c in range(len(self[0])):
            new[mainR][c] = self[brow][c]*bdelta + self[arow][c]*adelta
        return new

    def changeRow(self, arow: int, brow: int, other: Matrix) -> Matrix:
        new_a = self._copyArr()
        other_a = other._copyArr()
        new_a[arow] = other_a[brow]
        return Matrix(new_a, t=self.t)

    def changeCol(self, acol: int, bcol: int, other: Matrix) -> Matrix:
        return Matrix(_algebra.change_col(self.a, acol, bcol, other.a), self.t)

    def swapRow(self, arow: int, brow: int) -> Matrix:
        new_a = self._copyArr()
        new_a[arow], new_a[brow] = new_a[brow], new_a[arow]
        return Matrix(new_a, t=self.t)

    def T(self) -> Matrix:
        return Matrix(_algebra.transpose(self.a), self.t)

    def rot90(self) -> Matrix:
        return Matrix([r[::-1] for r in self.T()], self.t)

    @overload
    def concat(self, bmat: Matrix, in_place: Literal[False] = False) -> Matrix: ...

    @overload
    def concat(self, bmat: Matrix, in_place: Literal[True]) -> None: ...

    @overload
    def concat(self, bmat: Matrix, in_place: bool) -> Matrix | None: ...

    def concat(self, bmat: Matrix, in_place: bool = False) -> Matrix | None:
        if not in_place:
            return Matrix(_algebra.concat(self.a, bmat.a), self.t)
        if len(bmat) != len(self):
            assert False, "not matching size"
        for i in range(len(self)):
            self.a[i] += bmat[i]

    @staticmethod
    def im(n: int, t: ScalarParser = eval) -> Rows:
        return _algebra.im(n, t)

    @staticmethod
    def imat(n: int, t: ScalarParser = eval) -> Matrix:
        return Matrix(Matrix.im(n, t), t=t)

    @staticmethod
    def zero_vec(n: int, t: ScalarParser = eval) -> Matrix:
        return Matrix(([[t("0")] for _ in range(n)]), t=t)

    @staticmethod
    def mvec(*elements: Any, t: ScalarParser = eval) -> Matrix:
        return Matrix([[x] for x in elements], t=t)

    def _get(self) -> None:
        self.a = []
        n = int(input())
        for _ in range(n):
            self.a.append(list(map(self.t, input().split())))

    def __str__(self) -> str:
        s = ""
        for r in self:
            s += " ".join(map(str, r)) + "\n"
        return s

    def show(self) -> None:
        for r in self:
            print(*r)
        print()

    def print_l(self) -> None:
        print(str(self.a).replace(" ", ""))

    def det2d(self) -> Any:
        return _algebra.det2d(self.a)

    def det3d(self) -> Any:
        return _algebra.det3d(self.a, self.t)

    def sign_cof(self, row: int, col: int) -> int:
        return _algebra.sign_cof(row, col)

    def minor(self, row: int, col: int) -> Any:
        return _algebra.minor(self.a, row, col, self.t)

    def cof(self, row: int, col: int) -> Any:
        return _algebra.cof(self.a, row, col, self.t)

    def minorMat(self) -> Matrix:
        return Matrix(_algebra.minorMat(self.a, self.t), self.t)

    def cofMat(self) -> Matrix:
        return Matrix(_algebra.cofMat(self.a, self.t), self.t)

    def adj(self) -> Matrix:
        return Matrix(_algebra.adj(self.a, self.t), self.t)

    def det(self) -> Any:
        return _algebra.det(self.a, self.t)

    def inv_MIA(self) -> Matrix:
        return Matrix(_algebra.inv_MIA(self.a, self.t), self.t)

    def inv(self) -> Matrix:
        return Matrix(_algebra.inv(self.a, self.t), self.t)

    def inv2d(self) -> Matrix:
        return Matrix(_algebra.inv2d(self.a), self.t)

    @staticmethod
    def isinv(a: Matrix, b: Matrix) -> bool:
        if not isinstance(a, Matrix) or not isinstance(b, Matrix):
            return False
        return _algebra.isinv(a.a, b.a, a.t, b.t)

    def solve(self, b: Matrix) -> Matrix:
        return Matrix(_algebra.solve(self.a, b.a, self.t))

    def solveSelf(self) -> Matrix:
        return Matrix(_algebra.solveSelf(self.a, self.t))

    def isrref(self, arr: Rows | None = None) -> bool:
        return _algebra.isrref(self.a if arr is None else arr)

    def rref(self, tol: float = 1e-12) -> Matrix:
        rows, t = _algebra.rref(self.a, self.t, tol)
        return Matrix(rows, t)

    def col_space(self, tol: float = 1e-12) -> list[Matrix]:
        return [Matrix(rows, self.t) for rows in _algebra.col_space(self.a, self.t, tol)]

    def row_space(self, tol: float = 1e-12) -> list[Matrix]:
        basis, t = _algebra.row_space(self.a, self.t, tol)
        return [Matrix(rows, t) for rows in basis]

    def is_vector(self) -> bool:
        return _vectors.is_vector(self.a)

    def vR(self, pos: int) -> Any:
        return _vectors.vR(self.a, pos)

    def dot(self, other: Matrix) -> Any:
        return _vectors.dot(self.a, other.a)

    def cross(self, other: Matrix) -> Matrix:
        return Matrix(_vectors.cross(self.a, other.a), self.t)

    def cB(self, basis: list[Matrix]) -> Matrix:
        return Matrix(_vectors.cB(self.a, [vector.a for vector in basis], self.t))

    def in_span(self, basis: list[Matrix]) -> bool:
        return _vectors.in_span(self.a, [vector.a for vector in basis])

    def cA(self) -> Poly:
        return _spectral.cA(self.a, self.t)

    def eigen_vals(self) -> list[Any]:
        return _spectral.eigen_vals(self.a, self.t)

    def is_similar(self, other: Matrix) -> bool:
        return _spectral.is_similar(self.a, other.a, self.t, other.t)

    def eigen_vec(self, eigen_val: Any) -> Matrix:
        rows, t = _spectral.eigen_vec(self.a, eigen_val, self.t)
        return Matrix(rows, t)

    def algebraic_multiplicity(self, eigen_val: Any) -> int:
        return _spectral.algebraic_multiplicity(self.a, eigen_val, self.t)

    def geometric_multiplicity(self, eigen_val: Any) -> int:
        return _spectral.geometric_multiplicity(self.a, eigen_val, self.t)

    def is_diagnolizable(self) -> bool:
        return _spectral.is_diagnolizable(self.a, self.t)

    def diag(self) -> Matrix:
        return Matrix(_spectral.diag(self.a, self.t), self.t)

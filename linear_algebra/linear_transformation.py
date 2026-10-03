from __future__ import annotations

from collections.abc import Callable

from linear_algebra.matrix import Matrix


class LinearTransformation:
    def __init__(self, input_dim: int, output_dim: int, transform: Callable[[Matrix], Matrix]):
        if input_dim < 1 or output_dim < 1:
            raise ValueError("Transformation dimensions must be positive.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.transform = transform

    def __call__(self, vector: Matrix) -> Matrix:
        if vector.shape != (self.input_dim, 1):
            raise ValueError("Input must be a column vector with the input dimension.")
        result = self.transform(vector)
        if not isinstance(result, Matrix) or result.shape != (self.output_dim, 1):
            raise ValueError("Output must be a column vector with the output dimension.")
        return result

    @staticmethod
    def standard_basis(size: int) -> list[Matrix]:
        identity = Matrix.identity(size)
        return [Matrix.vector(*row) for row in identity]

    def matrix(self) -> Matrix:
        return Matrix.from_cols([self(vector) for vector in self.standard_basis(self.input_dim)])

    def matrix_in_bases(self, input_basis: list[Matrix], output_basis: list[Matrix]) -> Matrix:
        source = Matrix.from_cols(input_basis)
        target = Matrix.from_cols(output_basis)
        if source.shape != (self.input_dim, self.input_dim) or source.rank() != self.input_dim:
            raise ValueError("Input basis must be a full independent basis.")
        if target.shape != (self.output_dim, self.output_dim) or target.rank() != self.output_dim:
            raise ValueError("Output basis must be a full independent basis.")
        return Matrix.from_cols([self(vector).coords(output_basis) for vector in input_basis])

    def inverse(self) -> LinearTransformation:
        inverse_matrix = self.matrix().inv()
        return LinearTransformation(self.output_dim, self.input_dim, lambda vector: inverse_matrix @ vector)

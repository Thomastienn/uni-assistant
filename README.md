# uni-assistant

Small learning tools for linear algebra and logic.

## Setup

```bash
python -m pip install -r requirements.txt
python sample_main.py
```

## Linear algebra

Matrix algorithms live here so you can read and change them. SymPy supplies
polynomials, polynomial roots, and scalar arithmetic for exact roots such as
`sqrt(2)`. No SymPy matrices, matrix solvers, or eigenvector routines are used.

Read the code in this order:

| File | What to learn |
| --- | --- |
| [matrix.py](linear_algebra/matrix.py) | Public API, storage, operators, elementary row operations |
| [_matrix_algebra.py](linear_algebra/_matrix_algebra.py) | Matrix multiplication, cofactor determinant, scalar arithmetic |
| [_row_reduction.py](linear_algebra/_row_reduction.py) | Pivot selection, RREF, solving, inverse, rank, spaces |
| [_matrix_vectors.py](linear_algebra/_matrix_vectors.py) | Dot/cross products, coordinates, orthogonality, projection |
| [_matrix_spectral.py](linear_algebra/_matrix_spectral.py) | Characteristic polynomial, eigenspaces, diagonalization, similarity |
| [polynomial.py](linear_algebra/polynomial.py) | Native SymPy `Poly`, `x`, and reduction modulo a root's polynomial |
| [sample_main.py](sample_main.py) | Runnable example with assertions |

### Matrices and vectors

```python
from fractions import Fraction
from linear_algebra.matrix import Matrix

a = Matrix([[2, 1], [1, 2]])
b = Matrix([[1, 0], [0, 3]])
v = Matrix.vector(3, 1)

a.shape                    # (2, 2)
a[0][1]                    # 1; indices start at zero
a.data                     # Underlying list of rows
print(a)                   # Rows without brackets
a.copy()                   # Independent row lists
a.astype(float)            # New matrix with converted entries
Matrix.identity(3)
Matrix.zeros(3, 2)
Matrix.from_columns([Matrix.vector(1, 0), Matrix.vector(0, 1)])

a + b
a - b
a @ b                      # Matrix multiplication
3 * a                      # Scalar multiplication
a / 2                      # Exact Fraction entries for integer inputs
a ** 3                     # Integer powers; negative powers use inverse()
a.transpose()
a.rotate90()
a | v                      # Same as a.augment(v)
a == b                     # Exact entry equality
a.is_close(b)              # Numerical tolerance or exact scalar simplification
```

Construction copies the supplied rows. Methods return new matrices; they do not
change their inputs. Direct writes such as `a[0][1] = 7` still change `a`;
keep rows rectangular and entries numeric when editing `data`.

Vectors and solution vectors are columns. Space methods return lists of column
vectors, including `row_space()`, which represents each basis row as a column.
The empty basis is `[]`; `Matrix([])` represents a 0 by 0 matrix. Other
zero-dimension shapes are not supported.

For keyboard input, use `Matrix.from_input()`: first enter the row count, then
one space-separated row per line. Its default parser is `Fraction`, so
`1/3` stays exact. Use `Matrix.from_input(float)` or
`Matrix.from_input(complex)` for other numeric input.

### Row operations and systems

```python
a.swap_rows(0, 1)
a.swap_columns(0, 1)
a.combine_rows(target=1, source=0, source_scale=-2)  # R1 <- R1 - 2 R0
a.without_row(0)
a.without_column(0)

a.det()
a.minor(0, 1)
a.cofactor(0, 1)
a.cofactor_matrix()
a.adjugate()
a.inverse()
a.is_inverse_of(a.inverse())

a.rref()
a.is_rref()
a.rank()
a.null_space()
a.column_space()
a.row_space()

rhs = Matrix.vector(1, 0)
solution = a.solve(rhs)     # Column vector [2/3, -1/3]
assert a @ solution == rhs
assert (a | rhs).solve_augmented() == solution
```

The shared `reduce_rows()` function returns both RREF and pivot columns.
`solve()` reduces `[A | b]`; a pivot in the right-hand side means an
inconsistent system, while missing coefficient pivots mean infinitely many
solutions. Both cases raise `ValueError`. For a homogeneous system, use
`null_space()` to get a basis instead.

Multiple right-hand sides work too: `A.solve(B)` returns `X` with
`A @ X == B`. Inverse uses this same algorithm with `B = I`.
Determinant uses cofactor expansion so the formula is visible; its factorial
cost makes it suitable for small learning examples.

Integer and Fraction elimination stays exact. Floats use partial pivoting and
the absolute threshold `linear_algebra.config.TOLERANCE` (default `1e-12`).
Use exact inputs when rank or multiplicity matters. Scalar symbolic expressions
can be used in determinants; substitute numerical values before row reduction
or spectral calculations if they contain free symbols.

### Coordinates and projection

```python
basis = [Matrix.vector(1, 1), Matrix.vector(1, -1)]
v = Matrix.vector(3, 1)

v.dot(Matrix.vector(1, 2))       # 5
Matrix.vector(1, 0, 0).cross(Matrix.vector(0, 1, 0))
v.coordinates(basis)            # Column vector [2, 1]
v.in_span(basis)                # True, also accepts dependent spanning sets
Matrix.is_orthogonal(basis)     # True
v.orthogonal_coordinates(basis)
v.project(Matrix.vector(1, 1))  # Column vector [2, 2]
```

`coordinates()` solves the system whose columns are the supplied independent
basis vectors. It also works for a basis of a proper subspace, provided the
target is in that subspace.

Orthogonal coordinates use `(v dot f) / (f dot f)`. If the target is outside
the supplied orthogonal span, these are its projection coefficients.
Orthogonality and projection require real vectors. `dot()` is the bilinear
sum of entry products; it does not conjugate complex entries.

### Polynomials and eigenvectors

```python
from linear_algebra.polynomial import Poly, x

p = Poly(x**3 - 4*x**2 + 5*x - 2, x)
p.all_roots()               # [1, 1, 2], includes multiplicity
p.all_coeffs()              # [1, -4, 5, -2], highest degree first
p.eval(2)                  # 0
p.diff()                   # Derivative as another Poly
p.as_expr()                # Polynomial expression
p.div(Poly(x - 1, x))       # (quotient, remainder)
```

`Poly` is SymPy's class directly. Write `x**2`, not `x^2`.
For matrix entries, use scalar expressions such as `x + 1` or
`p.as_expr()`, rather than `Poly` objects.
See [SymPy's polynomial reference](https://docs.sympy.org/latest/modules/polys/reference.html).

```python
a = Matrix([[2, 1], [1, 2]])
a.charpoly()                # Poly(x**2 - 4*x + 3, x)
a.eigenvalues()             # [1, 3]
a.eigenspace(1)             # [Matrix([[-1], [1]])]
a.algebraic_multiplicity(1) # Root multiplicity in charpoly
a.geometric_multiplicity(1) # Dimension of eigenspace

p, d = a.diagonalize()
assert (a @ p).is_close(p @ d)
assert (p @ d @ p.inverse()).is_close(a)
```

Follow these steps in `_matrix_spectral.py`:

1. Form `x*I - A` using our matrix arithmetic.
2. Compute its determinant using our cofactor algorithm.
3. Ask SymPy for roots of that polynomial.
4. For each root `lambda`, compute our `null_space(A - lambda*I)`.
5. Put independent eigenvectors into columns of `P`; put matching eigenvalues
   onto the diagonal of `D`.

Eigenvalues are not rounded. Exact integer/Fraction inputs can yield rational
numbers, radicals, or SymPy `CRootOf` objects. Such objects represent exact
polynomial roots even when simple radical formulas are unavailable.
Scalar simplification also uses SymPy; pivot selection and elimination stay
in our code. For exact algebraic entries, row reduction uses SymPy polynomial
fields for scalar arithmetic: powers of a root reduce modulo its polynomial.
Complicated combinations of algebraic roots can still be slow.

These spectral methods work over the complex numbers. For example, a real
rotation matrix can have complex eigenvectors and still be diagonalizable.
`eigenspace()` returns `[]` for a number that is not an eigenvalue.
`diagonalize()` raises `ValueError` without a full eigenvector basis.

`is_similar()` checks both the characteristic polynomial and the ranks of
successive powers of `A - lambda*I`. Those ranks distinguish Jordan block
sizes, so matching eigenvalues alone no longer produces a false positive.
Polynomial root finding here targets numerical algebraic coefficients;
SymPy may reject polynomials with transcendental coefficients.

### Linear transformations

```python
from linear_algebra.linear_transformation import LinearTransformation

transform = LinearTransformation(2, 2, lambda v: a @ v)
transform(Matrix.vector(1, 0))
transform.matrix()
transform.matrix_in_bases(basis, basis)
transform.inverse()(Matrix.vector(1, 0))
LinearTransformation.standard_basis(2)
```

The supplied function must be linear. Dimensions are checked; sampling a few
inputs cannot prove an arbitrary Python function is linear.

### Migration from the old API

This redesign intentionally changes names and some return values.

| Old | New |
| --- | --- |
| `Matrix()`, `Matrix(t=Fraction)` | `Matrix.from_input()` |
| `a.a` | `a.data` |
| `a * b` for two matrices | `a @ b` |
| `a.T()`, `a.rot90()` | `a.transpose()`, `a.rotate90()` |
| `Matrix.imat(n)`, `Matrix.im(n)` | `Matrix.identity(n)`, `Matrix.identity(n).data` |
| `Matrix.mvec(...)`, `Matrix.zero_vec(n)` | `Matrix.vector(...)`, `Matrix.zeros(n, 1)` |
| `a._copyMat()`, `a._copyArr()` | `a.copy()`, `a.copy().data` |
| `a.show()`, `a.print_l()` | `print(a)`, `print(a.data)` |
| `a.concat(b)` | `a.augment(b)` or `a | b` |
| `a.swapRow(i, j)`, `a.assignRow(...)` | `a.swap_rows(i, j)`, `a.combine_rows(...)` |
| `a.removeRow(i)`, `a.removeCol(j)` | `a.without_row(i)`, `a.without_column(j)`; return copies |
| `a.cof(i, j)`, `a.cofMat()`, `a.adj()` | `a.cofactor(i, j)`, `a.cofactor_matrix()`, `a.adjugate()` |
| `a.inv()`, `a.inv_MIA()`, `a.inv2d()` | `a.inverse()`; shared exact row reduction |
| `Matrix.isinv(a, b)` | `a.is_inverse_of(b)` |
| `a.solve(b)`, `a.solveSelf()` | `a.solve(b)`, `a.solve_augmented()`; return columns |
| `a.isrref()`, `a.col_space()` | `a.is_rref()`, `a.column_space()` |
| `v.vR(i)` | `v[i][0]` |
| `v.cB(B)`, `v.cB_ortho(B)` | `v.coordinates(B)`, `v.orthogonal_coordinates(B)` |
| `Matrix.is_ortho(B)`, `v.proj(u)` | `Matrix.is_orthogonal(B)`, `v.project(u)` |
| `a.cA()`, `a.eigen_vals()` | `a.charpoly()`, `a.eigenvalues()` |
| `a.eigen_vec(value)` | `a.eigenspace(value)`; returns a basis, not an augmented RREF |
| `a.is_diagnolizable()`, `a.diag()` | `a.is_diagonalizable()`, `a.diagonalize()`; returns `(P, D)` |
| Custom `Poly.coef`, `Poly.val()`, `Poly.deriv()`, `Poly.solve()` | Native `all_coeffs()` (descending), `eval()`, `diff()`, `all_roots()` |

The separate `Vector` and `Vec3d` classes retain their existing APIs.

## Logic 
<details>
 
### Usage
``` python3
  # Declare a function
  def f1(x1,x2,x3):
    return x1 and not(x2 or x3)
  def f2(x1,x2,x3):
    return x1 or (x2 and not x3)

  # Check if 2 functions have same boolean for all cases
  checkEqual(f1, f2, 3) # 3 is the number of variables used in f1 and f2

  # Generate truth tables for function
  generateTable(f1, 3)                         # 3 is the number of variables used in f1
  generateTable(f1, 3, labels=["a", "b", "c"]) # Set the labels at the top of row

  # More examples of custom function
  # Can return multiple outputs 0 or 1
  def half_adder(x1, x2):
      summ = x1 ^ x2
      carry = x1 and x2 
      return summ, carry
```
</details>

# Purpose
This is my database of projects that supported me alot during classes in uni. I hope it can help you too. 
## Menu
* [LinearAlgebra](#linear-algebra)
* [Logic](#logic)

# Linear Algebra
<details>
 
### Usage
* Input need first number is the number of rows. After that is the matrix with space between
* Every function is returning a new copy matrix or a value.
* The function is using 0-indexed (use with caution)

**Input:**
```
  2
  1 2
  3 4
  2
  0 3
  4 5
  2
  2 x
  5x^2 10
```
**Samples**
``` python3
  # Get the matrix from stdin
  # You can do t=Fraction for precision (Poly is not compatible with Fraction yet)
  # t = float, t=int
  a = Matrix()
  b = Matrix()
  c = Matrix(t=Poly) # Need to use for poly class to use variables

  # Specify the type
  a = Matrix(t=Fraction)
  a = Matrix(t=int)
  a = Matrix(t=float)
  a = Matrix(t=complex)

  # Intialize from 2D arrays
  arr = [[1,2], [3,4]]
  a = Matrix(a=arr)

  # Show the array
  # Expected output:
  # 1 2
  # 3 4
  # [[1,2],[3,4]]
  a.show()    # Show the matrix with spacing and no brackets
  a.print_l() # Show as a 2D array

  # Operations (Return a new matrix)
  a * b  # Multiply 2 matrices
  a * 2  # Apply to all elements (Haven't support the left multiply -> You can't do 2 * a)
  a + b  # Add 2 matrices
  a - b  # Subtract 2 matrices
  a**10   # Matrix exponential (Use binary exponential) -> FAST
  a == b # Check if 2 matrices are identical

  # Get copies
  a._copyArr() # Get the copy of 2D list of matrix
  a._copyMat() # Get the new copy of current matrix

  # Assigning rows
  # b_delta, b_row, a_delta, a_row, applied row
  # used for RREF and other algorithms
  a.assignRow(2, 0, 1, 1, 1) # This means apply to -> row 1 = 2*(row 0) + 1*(row 1) (Zero indexed)
  a.swapRow(0,1)             # Swap row 0 and 1
  a.swapCol(0,1)             # Swap column 0 and 1 (In future update)
  a.removeRow(0)             # Remove row 0
  a.removeCol(1)             # Remove column 1

  # Special operations
  a.T()       # Get the transpose
  a.inv()     # Get the inverse of A if exist (Updated: You can switch between Matrix Inversion Algorithm or inverse by Determinant And Adjungate)
  a.rot90()   # Rotate clockwise 90 degrees
  a.concat(b) # Connect 2 matrices to create augmented matrix (Concat sideway) (in_place=False by default)
  a.rref()    # Get the Reduced Row Echelon Form of the matrix

  # Static, global methods
  Matrix.isinv(a, b)       # Check if 2 matrices are inverses
  Matrix.im(10)            # Create an identical matrix with type array of size n
  Matrix.imat(10)          # Return an identical matrix with type Matrix
  Matrix.mvector([1,2,3])  # Create a vertical vector (very convenient for linear transformation and spectral)
  Matrix.zero_vec()        # Get the zero vector

  # Checking
  a.isrref()                  # check if this matrix is in Reduced Row Echelon Form
  a.isrref(arr=[[0,1],[0,0]]) # check another array

  # Determinant, minor, cofactor
  a.minor(1, 3)   # Minor of the matrix if remove row 2 and column 4
  a.cof(1,3)      # Get the cofactor of this minor matrix by multiply with corresponding sign
  a.sign_cof(1,3) # You can get the sign using this function
  a.cofMat()      # Give you a whole new matrix, which entry (i,j) is cofactor of the minor matrix (i,j)
  a.adj()         # Give you the adjungate matrix
  a.det2d()       # Find determinant of 2d matrix (Must strictly 2d)
  a.det3d()       # Find determinant of 3d matrix (Use Sarrus rule)
  a.det()         # A more general use. Find determinant of matrix (any size) (Can switch between using Cofactor Method or Adjungate Method)
  
  a.solve(Matrix(a=[[1],[2],[3]]))       # Use Cramers' Rule to solve

  # Linear transformation
  a.cA()                               # Characteristic Polynomial
  a.vR(0)                              # For vertical vector, get the value at the "pos" row
  a.is_vector()                        # Help to check if it is vertical vector
  a.cB([Matrix(), Matrix(), Matrix()]) # Get coordinate vector according to a list of basis
  a.eigen_values()                     # Get all eigen values of this matrix

  # EXAMPLE SETUP
  def func(vec: Matrix):
     row1 = [vec.vR(0)+2*vec.vR(1)]
     row2 = [vec.vR(0) - vec.vR(1)]
     return Matrix(a=[row1, row2])
  lt = LinearTransformation(2, 2, func)
  B = [Matrix(), Matrix()]
  D = [Matrix(), Matrix()]
  # END

  # Static methods
  LinearTransformation.get_standard_basis()  # A generator for you to get standard basis in Rn (Ex: for basis in LinearTransformation.get_standard_basis())

  lt.get_transform_mat()                     # Get matrix A (the transform matrix of the linear transformation)
  lt.transform(Matrix(a=[[1],[2]])           # Transform a vector
  lt.get_transform_ADB(B,D)                  # Get the transform matrix A (from Rn to Rm where B is the basis in Rn and D is the basis in Rm)
  lt.get_inverse_transform_func()            # You can get the inverse function of the linear transformation from this
  lt.inv_transform(Matrix(a=[[1],[2]]))      # This one is simimlar to transform but using inverse function

  # Spectral theory
  a.is_similar(Matrix(a=[[1],[2],[3]]))      # In progress, not guarantee similar now
  a.eigen_vec(5)                             # Give you the rref solution of the eigen vector from corresponding eigen value
  a.algebraic_multiplicity(5)                # Give you the algebraic multiplicity of the eigen value
  a.geometric_multiplicity(5)                # Give you the geometric multiplicity of the eigen value
  a.is_diagnolizable()                       # Can this matrix be diagonalized
  a.diag()                                   # Get the matrix D (the diagonal matrix)
```
</details>

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

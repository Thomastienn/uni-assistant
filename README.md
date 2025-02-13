# Purpose
This is my database of projects that supported me alot during classes in uni. I hope it can help you too. 
* Matrix is created in order to speed up process in my linear algebra class.
* Logic is created in my discrete math class.
## Menu
* [Matrix](#matrix)
* [Logic](#logic)
* [Complex](#complex)
* [Quadratic Equation](#quadratic-equation)
## Matrix
### Usage
* Input need first number is the number of rows. After that is the matrix with space between
 
**Input:**
```
  2
  1 2
  3 4
  2
  0 3
  4 5
```
**Samples**
``` python3
  # Get the matrix from stdin
  a = Matrix()
  b = Matrix()

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
` a.print_l() # Show as a 2D array

  # Operations (Return a new matrix)
  a * b  # Multiply 2 matrices
  a * 2  # Apply to all elements
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
  a.assignRow(2, 0, 1, 1, 1) # This means apply to | row 1 = 2*(row 0) + 1*(row 1)

  # Special operations
  a.T()       # Get the transpose
  a.inv()     # Get the inverse of A if exist
  a.rot90()   # Rotate clockwise 90 degrees
  a.concat(b) # Connect 2 matrices to create augmented matrix (Concat sideway)
  a.rref()    # Get the Reduced Row Echelon Form of the matrix

  # Static, global methods
  Matrix.isinv(a, b) # Check if 2 matrices are inverses
  im(10)             # Create an identical matrix with type array of size n
  imat(10)           # Return an identical matrix with type Matrix

  # Checking
  a.isrref()                  # check if this matrix is in Reduced Row Echelon Form
  a.isrref(arr=[[0,1],[0,0]]) # check another array
```

## Logic 
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
## Complex
## Quadratic Equation

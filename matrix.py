from collections import deque
from fractions import Fraction

class Matrix:
    def __init__(self, t=eval, a=None):
        self.t = t
        if a is None:
            self.get()
        else:
            self.a = a
    def __mul__(self, bmat):
        ans = None
        if isinstance(bmat, Matrix):
            if len(self.a[0]) != len(bmat.a):
                assert False, "not matching size"
            ans = [[self.t("0")]*len(bmat.a[0]) for _ in range(len(self.a))]
            for i in range(len(self.a)):
                for j in range(len(bmat.a[0])):
                    for k in range(len(bmat.a)):
                        ans[i][j] += self.a[i][k]*bmat.a[k][j] 
        else:
            ans = [[0]*len(self.a[0]) for _ in range(len(self.a))]
            for i in range(len(self.a)):
                for j in range(len(self.a[0])):
                    ans[i][j] = self.a[i][j]*bmat
                    
        return Matrix(self.t, ans)
        
    def __eq__(self, b):
        a = self.a
        if len(a) != len(b) or len(a[0]) != len(b[0]):
            return False
        for i in range(len(a)):
            for j in range(len(a[0])):
                if a[i][j] != b[i][j]:
                    return False
    def __add__(self, bmat:"Matrix"):
        return self.add(bmat, 1)
    def __sub__(self, bmat:"Matrix"):
        return self.add(bmat, -1)
        
    def __pow__(self, n):
        amat = self._copyMat()
        iden = Matrix(a=self.im(len(self.a)))
        while n:
            if n&1:
                iden *= amat
            amat *= amat
            n >>= 1
            
        return iden
        
    def removeRow(self, row):
        self.a.pop(row)
        
    def removeCol(self,col):
        n = len(self.a)
        for i in range(n):
            self.a[i].pop(col)
    
    def det2d(self):
        n,m = len(self.a), len(self.a[0])
        assert (n==2 and m==2)
        
        return self.a[0][0]*self.a[1][1] - \
                self.a[0][1]*self.a[1][0]
    
    # Sarrus's Rule
    def det3d(self):
        assert len(self.a)==len(self.a[0])==3
        s = self.t("0")
        for off in range(3):
            pro1 = self.t("1")
            pro2 = self.t("1")
            for i in range(3):
                pro1 *= self.a[i][(i+off)%3]
                pro2 *= self.a[i][(off-i)%3]
            s += pro1 - pro2
        return s
    
    def sign_cof(self, row,col):
        return -1 if (row+col)&1 else 1
    
    def minor(self, row, col):
        n, m = len(self.a), len(self.a[0])
        new_a = []
        for i in range(n):
            for j in range(m):
                if i == row or j == col:
                    continue
                if not new_a or len(new_a[-1]) == m-1:
                    new_a.append([])
                new_a[-1].append(self.a[i][j])
        new_mat = Matrix(a=new_a)
        return new_mat.det()
        
        
    def cof(self,row,col):
        return self.sign_cof(row,col)*self.minor(row,col)
        
    def minorMat(self):
        new = [[0]*len(self.a[0]) for _ in range(len(self.a))]
        for i in range(len(self.a)):
            for j in range(len(self.a[0])):
                new[i][j] = self.minor(i,j)
        return Matrix(a=new)
        
    def cofMat(self):
        new = [[0]*len(self.a[0]) for _ in range(len(self.a))]
        for i in range(len(self.a)):
            for j in range(len(self.a[0])):
                new[i][j] = self.cof(i,j)
        return Matrix(a=new)
        
    def adj(self):
        return self.cofMat().T()
    
    # Naive method
    # def det(self):
        # n,m = len(self.a), len(self.a[0])
        # assert n == m
        # if n == 2:
            # return self.det2d()
        # ans = 0
        # # always first row    
        # for i in range(m):
            # ans += self.a[0][i] * self.cof(0,i)
        # return ans
    
    # Using adjugate
    def det(self):
        if len(self.a) == 0:
            return 1
        n,m = len(self.a), len(self.a[0])
        assert n == m
        if n == 2:
            return self.det2d()
        if n == 3:
            return self.det3d()
        return (self*self.adj()).a[0][0]
    
    def _copyArr(self):
        new = [[-1]*len(self.a[0]) for _ in range(len(self.a))]
        for i in range(len(self.a)):
            for j in range(len(self.a[0])):
                new[i][j] = self.a[i][j]
        return new
        
    def _copyMat(self):
        return Matrix(a=self._copyArr())
        
    def assignRow(self, bdelta, brow, adelta, arow, mainR):
        new = self._copyMat()
        for c in range(len(self.a[0])):
            new.a[mainR][c] = self.a[brow][c]*bdelta + self.a[arow][c]*adelta
        return new
    
    # Using cramer's rule
    def solve(self, b):
        detA = self.det()
        new_a = [Fraction(self.changeCol(i,0,b).det(),detA) for i in range(len(self.a))]
        return Matrix(a=[new_a])
        
    def changeRow(self,arow,brow,other):
        new_a = self._copyArr()
        other_a = other._copyArr()
        new_a[arow] = other_a[brow]
        return Matrix(a=new_a)
        
    def changeCol(self,acol,bcol,other):
        new_a = self._copyArr()
        other_a = other._copyArr()
        for i in range(len(self.a)):
            new_a[i][acol] = other_a[i][bcol]
        return Matrix(a=new_a)
    
    def swapRow(self, arow,brow):
        new_a = self._copyArr()
        new_a[arow], new_a[brow] = new_a[brow], new_a[arow]
        return Matrix(a=new_a)
        
        
    def get(self):
        self.a = []
        n = int(input())
        for _ in range(n):
            self.a.append(list(map(self.t, input().split())))
            
        
    def add(self, bmat: "Matrix", delta=1):
        b = bmat.a
        if len(self.a) != len(b) or len(self.a[0]) != len(b[0]):
            return None
        ans = [[0]*len(self.a[0]) for _ in range(len(self.a))]
        for i in range(len(self.a)):
            for j in range(len(self.a[0])):
                ans[i][j] = (self.a[i][j] + (b[i][j]*delta))
                
        return Matrix(self.t, ans)
        
    
    def T(self):
        return Matrix(self.t, [list(r) for r in zip(*self.a)])
        
    # LLM WORK
    # Matrix Inversion Algorithm
    # def inv(self): 
        # A = [row[:] for row in self.a]
        # I = [[1 if i == j else 0 for j in range(len(self.a[0]))] for i in range(len(self.a))]
        # for i in range(len(self.a)): 
            # pivot = A[i][i]
            # if pivot == 0: 
                # for j in range(i + 1, len(self.a)): 
                    # if A[j][i] != 0:
                        # A[i], A[j] = A[j], A[i] 
                        # I[i], I[j] = I[j], I[i] 
                        # pivot = A[i][i] 
                        # break 
            # if pivot == 0: 
                # raise ValueError("Matrix is singular and cannot be inverted.") 
            # for j in range(len(self.a)): 
                # A[i][j] /= pivot 
                # I[i][j] /= pivot 
            # for j in range(len(self.a)): 
                # if i != j: 
                    # factor = A[j][i] 
                    # for k in range(len(self.a)): 
                        # A[j][k] -= factor * A[i][k] 
                        # I[j][k] -= factor * I[i][k] 
        # return Matrix(type(I[0][0]), I)
    
    # Inverse by determinant and adjungate
    def inv(self):
        if len(self.a) == 0:
            return Matrix(a=[])
        assert len(self.a) == len(self.a[0]), "Must be a square"
        if len(self.a) == 1:
            assert self.a[0][0] != 0, "No inverse"
            return Matrix(a=[[Fraction(1,self.a[0][0])]])
        if len(self.a) == 2:
            return self.inv2d()
        new_mat = self._copyMat()
        return new_mat.adj()*(1/new_mat.det())
        
    def inv2d(self):
        new_mat = Matrix(a=[[self.a[1][1], -self.a[0][1]], [-self.a[1][0], self.a[0][0]]])
        return new_mat*(1/(self.a[0][0]*self.a[1][1] - self.a[0][1]*self.a[1][0]))
        
        
    def rot90(self):
        return Matrix(self.t,[r[::-1] for r in self.T(self.a)])
        
    def isinv(a:"Matrix", b:"Matrix"):
        return a*b == b*a
        
    def show(self):
        for r in self.a:
            print(*r)
        print()
        
    def print_l(self):
        print(str(self.a).replace(" ", ""))
        
    def _rowNonZero(self, rows):
        for i in range(len(rows)):
            if rows[i] != 0:
                return i
        return len(rows)
        
    def _rearrange(self,arr):
        arr.sort(key=lambda r: self._rowNonZero(r))
        
    def isrref(self, arr=None):
        if arr is None:
            arr = self._copyArr()
        prev = None
        for r in range(len(arr)):
            i = self._rowNonZero(arr[r])
            
            # FULL OF ZEROS
            if i == len(arr[r]) and r != len(arr)-1:
                return False
                
            # BELOW AND ABOVE MUST BE 0 TOO
            for r1 in range(r+1, len(arr)):
                if arr[r1][i] != 0:
                    return False
            for r1 in range(r):
                if arr[r1][i] != 0:
                    return False
                    
            # THE LAST ONE MUST BE MORE LEFT
            if prev != None and i <= prev:
                return False
            prev = i
        
        return True
        
    def concat(self, bmat):
        if len(bmat.a) != len(self.a):
            assert False, "not matching size"
        new = self._copyArr()
        for i in range(len(new)):
            new[i] += bmat.a[i]
        
        return Matrix(a=new)
    
    # LLM WORK 
    def rref(self):
        A = self._copyArr()
        rows, cols = len(A), len(A[0])
        r = 0  # Row index
    
        for c in range(cols):
            # Find the row with the largest absolute value in column c
            pivot_row = max(range(r, rows), key=lambda i: abs(A[i][c]), default=None)
            if pivot_row == None or A[pivot_row][c] == 0:
                continue  # Skip if column is all zeros
    
            # Swap the current row with the pivot row
            A[r], A[pivot_row] = A[pivot_row], A[r]
    
            # Normalize the pivot row (make leading coefficient 1)
            pivot = A[r][c]
            A[r] = [val / pivot for val in A[r]]
    
            # Eliminate all other entries in the current column
            for i in range(rows):
                if i != r:
                    factor = A[i][c]
                    A[i] = [A[i][j] - factor * A[r][j] for j in range(cols)]
    
            r += 1  # Move to the next row
    
        return Matrix(a=A,t=self.t)

def im(n):
        return [[1 if i == j else 0 for j  in range(n)] for i in range(n)]
def imat(n):
    return Matrix(a=im(n))



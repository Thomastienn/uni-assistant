import math
from fractions import Fraction
class Complex:
    def __init__(self, r,i,t=float):
        self.r = t(r)
        self.i = t(i)
        self.d = t((r**2+i**2)**0.5)
    def _to_complex(self,com):
        if isinstance(com, int):
            com = Complex(com, 0)
        return com
    def conj(self):
        return Complex(self.r, -self.i)
    def arg(self):
        # cosx = a/r 
        # sinx = b/r
        x = math.acos(self.r/self.d)
        return x if self.i > 0 else -x
    
    def polar(self):
        return self.d, self.arg()
    
    def __add__(self, other):
        other = self._to_complex(other)
        return Complex(self.r+other.r, self.i+other.i)
    def __sub__(self, other):
        other = self._to_complex(other)
        return Complex(self.r-other.r, self.i-other.i)
    def __mul__(self, other):
        other = self._to_complex(other)
        r = self.r*other.r - self.i*other.i
        i = self.r*other.i + self.i*other.r
        
        return Complex(r, i)
    def __truediv__(self, other):
        other = self._to_complex(other)
        # (a+bi)/(c+di) = (a+bi)(c-di)/(c+di)(c-di)
        # c^2 - d^2 * (-1)
        nume = self*other.conj()
        fac = other.r**2 + other.i**2
        return Complex(nume.r/fac,nume.i/fac)
    def __repr__(self):
        if self.r == 0 and self.i == 0:
            return "0"
        real = "" if self.r == 0 else self.r
        sign = "+" if (self.i > 0 and real) else ("-" if self.i < 0 else "")
        imag = "" if self.i == 0 else str(abs(self.i))+"i"
        return f"{real}{sign}{imag}"
    def __pow__(self, n):
        ans = Complex(1, 0)
        cur = Complex(self.r, self.i)
        while n:
            if n&1:
                ans *= cur
            cur *= cur
            n >>= 1
        return ans
        
class QuadEq:
    def __init__(self,a,b,c):
        self.a = a
        self.b = b
        self.c = c
        self.disc = b**2 - 4*a*c
        
    def solve(self):
        if self.a == 0:
            return Fraction(-self.c,self.b)
        if self.disc == 0:
            return Fraction(-self.b,2*a)
        if self.disc > 0:
            return -self.b+math.sqrt(self.disc)/2*self.a, -b-math.sqrt(self.disc)/2*self.a
        
        return Complex(-self.b, math.sqrt(abs(self.disc)))/(2*self.a),\
            Complex(-self.b, -math.sqrt(abs(self.disc)))/(2*self.a)
    def __repr__(self):
        x2 = (str(self.a) if abs(self.a) != 1 else "") + "x^2" if self.a != 0 else ""
        sign1 = "-" if self.b < 0 else ("+" if x2 and self.b != 0 else "")
        x = (str(abs(self.b)) if abs(self.b) != 1 else "") + "x" if self.b != 0 else ""
        sign2 = "-" if self.c < 0 else ("+" if (x2 or self.b != 0) and self.c != 0 else "")
        c = str(abs(self.c)) if self.c != 0 else ""
        return f"{x2}{sign1}{x}{sign2}{c}"
        
def rad_deg(rad):
    return rad*180/math.pi
def deg_rad(deg):
    return deg*math.pi/180
    
a = Complex(-1, 1)

print(a.arg())



        
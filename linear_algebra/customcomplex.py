import math
from fractions import Fraction
class Complex:
    def __init__(self, r,i=0,t=Fraction):
        self.t = t
        if isinstance(r,str):
            self.r, self.i = self.parse(r)
        else:
            self.r = t(r)
            self.i = t(i)
        self.d = t((self.r**2+self.i**2)**0.5)
    def _to_complex(self,com):
        if isinstance(com, int):
            com = Complex(com, 0, t=self.t)
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
        return Complex(self.r+other.r, self.i+other.i, t=self.t)
    def __sub__(self, other):
        other = self._to_complex(other)
        return Complex(self.r-other.r, self.i-other.i,t=self.t)
    def __mul__(self, other):
        other = self._to_complex(other)
        r = self.r*other.r - self.i*other.i
        i = self.r*other.i + self.i*other.r
        
        return Complex(r, i,t=self.t)
    def __truediv__(self, other):
        other = self._to_complex(other)
        # (a+bi)/(c+di) = (a+bi)(c-di)/(c+di)(c-di)
        # c^2 - d^2 * (-1)
        nume = self*other.conj()
        fac = other.r**2 + other.i**2
        return Complex(nume.r/fac,nume.i/fac,t=self.t)
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
    def parse(self,s):
        s = s.replace(" ", "")
        if s in ("i", "+i"):
            return self.t(0), self.t(1)
        if s == "-i":
            return self.t(0), self.t(-1)
        if 'i' in s:
            if not s.endswith('i'):
                raise ValueError("Invalid format")
            s = s[:-1]
            r, i = "", ""
            idx = -1
            for j in range(1, len(s)):
                if s[j] in '+-':
                    idx = j
                    break
            if idx != -1:
                r, i = s[:idx], s[idx:]
                if i in ("+", "-"):
                    i += "1"
                if r == "":
                    r = "0"
            else:
                r, i = "0", s
                if i in ("+", "-"):
                    i += "1"
            return self.t(r), self.t(i)
        return self.t(s), self.t(0)

def rad_deg(rad):
    return rad*180/math.pi
def deg_rad(deg):
    return deg*math.pi/180


        
from fractions import Fraction
import math
class Poly:
    def __init__(self, coef, t=Fraction):
        self.t = t
        if isinstance(coef, str):
            self.coef = self.parse(coef)
        elif isinstance(coef, list):
            self.coef = coef
        else:
            assert False, "No Support"

    def __add__(self, other):
        if isinstance(other, int) or isinstance(other, Fraction):
            return Poly([self.coef[0]+other] + self.coef[1:])
        n, m = len(self.coef), len(other.coef)
        new = [0] * max(n,m)
        for i in range(n):
            new[i] += self.coef[i]
        for i in range(m):
            new[i] += other.coef[i]
        return Poly(new)

    def __sub__(self, other):
        if isinstance(other, int) or isinstance(other, Fraction):
            return Poly([self.coef[0]-other] + self.coef[1:])
        n, m = len(self.coef), len(other.coef)
        new = [0] * max(n,m)
        for i in range(n):
            new[i] += self.coef[i]
        for i in range(m):
            new[i] -= other.coef[i]
        return Poly(new)

    def __mul__(self, other):
        if isinstance(other, int) or isinstance(other, Fraction):
            return Poly([self.coef[i]*other for i in range(len(self.coef))])
        n, m = len(self.coef), len(other.coef)
        new = [0] * (n + m - 1)
        for i, c1 in enumerate(self.coef):
            for j, c2 in enumerate(other.coef):
                new[i + j] += c1 * c2
        return Poly(new)
        
    def val(self, x):
        return sum(c * (x ** i) for i, c in enumerate(self.coef))
    def deriv(self):
        new = [i * c for i, c in enumerate(self.coef)][1:]
        return Poly(new)
    def __str__(self):
        terms = []
        for i in range(len(self.coef) - 1, -1, -1):  # Iterate from highest degree down
            c = self.coef[i]
            if c == 0:
                continue
            if i == 0:
                terms.append(f"{c}")
            elif i == 1:
                terms.append(f"{'' if c == 1 else '-' if c == -1 else c}x")
            else: 
                terms.append(f"{'' if c == 1 else '-' if c == -1 else c}x^{i}")
        return "+".join(terms).replace("+-", "-") if terms else "0"
    def parse(self,poly_str):
        poly_str = poly_str.replace(" ", "").replace("-", "+-")
        terms = poly_str.split("+")
        coeffs = {}
        for term in terms:
            if not term:
                continue
            if "x" in term:
                if "^" in term:
                    coeff, exp = term.split("x^")
                else:
                    coeff, exp = term.split("x")
                    exp = "1"
                coeff = coeff.strip()
                if coeff in ["", "+", "-"]:
                    coeff += "1"
                coeff = self.t(coeff)  
                exp = int(exp)
            else:
                coeff = self.t(term)
                exp = 0
            coeffs[exp] = coeffs.get(exp, self.t(0)) + coeff
        max_degree = max(coeffs.keys(), default=0)
        return [coeffs.get(i, 0) for i in range(max_degree + 1)]
        
    def __abs__(self):
        return 0
        
    def __truediv__(self, other):
        if self == other:
            return Poly([1])
        if(isinstance(other, int)):
            other = Poly([other])
        if (len(other.coef) >  1 and len(self.coef) > 1):
            assert False, "Not support poly"
        if other.coef[0] == 0:
            assert False, "Zero division"
        new = []
        if len(self.coef) == 1:
            new = [Fraction(self.coef[0], other)]
        else:
            new = [Fraction(self.coef[i], other.coef[0]) for i in range(len(self.coef))]

        return Poly(coef=new)
        
    def __eq__(self, other):
        if isinstance(other, int):
            return (len(self.coef) == 1 and self.coef[0] == other)
        if len(self.coef) != len(other.coef):
            return False
        return all(self.coef[i] == other.coef[i] for i in range(len(self.coef)))
        
    def solve(self):
        if len(self.coef) == 1:
            return [] if self.coef[0] != 0 else [0]
        if len(self.coef) == 2:
            return [-self.coef[0] / self.coef[1]]
        if len(self.coef) == 3:
            a, b, c = self.coef[2], self.coef[1], self.coef[0]
            d = b**2 - 4*a*c
            if d > 0:
                return [(-b + math.sqrt(d)) / (2*a), (-b - math.sqrt(d)) / (2*a)]
            elif d == 0:
                return [-b / (2*a)]
            else:
                return [complex(-b, math.sqrt(-d)) / (2*a), complex(-b, -math.sqrt(-d)) / (2*a)]
        assert False, "Not support"
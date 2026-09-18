import cmath
from fractions import Fraction
from itertools import zip_longest
import math
from numbers import Number, Rational
import re


class Poly:
    def __init__(self, coef, t=Fraction):
        self.t = t
        if isinstance(coef, str):
            coefficients = self.parse(coef)
        elif isinstance(coef, (list, tuple)):
            coefficients = list(coef)
        elif isinstance(coef, Number):
            coefficients = [coef]
        else:
            raise TypeError("Expected a polynomial string, coefficient list, or number.")
        if any(not isinstance(value, Number) for value in coefficients):
            raise TypeError("Polynomial coefficients must be numbers.")
        self.coef = coefficients or [0]
        while len(self.coef) > 1 and self.coef[-1] == 0:
            self.coef.pop()

    def __add__(self, other):
        if isinstance(other, Number):
            other = Poly(other, t=self.t)
        if not isinstance(other, Poly):
            return NotImplemented
        coefficients = [a + b for a, b in zip_longest(self.coef, other.coef, fillvalue=0)]
        return Poly(coefficients, t=self.t)

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return Poly([-value for value in self.coef], t=self.t)

    def __sub__(self, other):
        if not isinstance(other, (Poly, Number)):
            return NotImplemented
        return self + (-other)

    def __rsub__(self, other):
        if not isinstance(other, (Poly, Number)):
            return NotImplemented
        return -self + other

    def __mul__(self, other):
        if isinstance(other, Number):
            return Poly([value * other for value in self.coef], t=self.t)
        if not isinstance(other, Poly):
            return NotImplemented
        coefficients = [0] * (len(self.coef) + len(other.coef) - 1)
        for i, a in enumerate(self.coef):
            for j, b in enumerate(other.coef):
                coefficients[i + j] += a * b
        return Poly(coefficients, t=self.t)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, Number):
            other = Poly(other, t=self.t)
        if not isinstance(other, Poly):
            return NotImplemented
        if other == 0:
            raise ZeroDivisionError("Cannot divide a polynomial by zero.")
        if self == other:
            return Poly(1, t=self.t)
        if len(other.coef) != 1:
            raise NotImplementedError("Division by a nonconstant polynomial is not supported.")
        divisor = other.coef[0]
        coefficients = [Fraction(value, divisor)
                        if isinstance(value, Rational) and isinstance(divisor, Rational)
                        else value / divisor for value in self.coef]
        return Poly(coefficients, t=self.t)

    def __eq__(self, other):
        if isinstance(other, Number):
            return len(self.coef) == 1 and self.coef[0] == other
        if not isinstance(other, Poly):
            return NotImplemented
        return self.coef == other.coef

    def val(self, x):
        result = 0
        for coefficient in reversed(self.coef):
            result = result * x + coefficient
        return result

    def deriv(self):
        return Poly([power * self.coef[power] for power in range(1, len(self.coef))], t=self.t)

    def __str__(self):
        terms = []
        for power in range(len(self.coef) - 1, -1, -1):
            coefficient = self.coef[power]
            if coefficient == 0:
                continue
            if power == 0:
                terms.append(str(coefficient))
                continue
            if coefficient == 1:
                prefix = ""
            elif coefficient == -1:
                prefix = "-"
            else:
                prefix = str(coefficient)
            variable = "x" if power == 1 else f"x^{power}"
            terms.append(prefix + variable)
        return "+".join(terms).replace("+-", "-") or "0"

    def parse(self, poly_str):
        expression = "".join(poly_str.split())
        if not expression:
            raise ValueError("Polynomial expression cannot be empty.")
        # A coefficient may be a decimal, scientific notation, or a fraction.
        number = r"(?:\d+/\d+|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
        term_pattern = re.compile(
            rf"([+-]?)(?:({number})?x(?:\^(\d+))?|({number}))"
        )
        coefficients = {}
        position = 0
        while position < len(expression):
            match = term_pattern.match(expression, position)
            if match is None or (position > 0 and not match.group(1)):
                raise ValueError(f"Invalid polynomial expression near {expression[position:]!r}.")
            sign, coefficient, power, constant = match.groups()
            if constant is not None:
                value, power = self.t(constant), 0
            else:
                value, power = self.t(coefficient or "1"), int(power or "1")
            if sign == "-":
                value = -value
            coefficients[power] = coefficients.get(power, 0) + value
            position = match.end()
        return [coefficients.get(power, 0) for power in range(max(coefficients) + 1)]

    def solve(self):
        degree = len(self.coef) - 1
        if degree == 0:
            if self.coef[0] == 0:
                raise ValueError("The zero polynomial has infinitely many roots.")
            return []
        if degree == 1:
            constant, slope = self.coef
            if isinstance(constant, Rational) and isinstance(slope, Rational):
                return [Fraction(-constant, slope)]
            return [-constant / slope]
        if degree != 2:
            raise NotImplementedError("Root solving supports only degree 0, 1, or 2.")

        c, b, a = self.coef
        discriminant = b * b - 4 * a * c
        if discriminant == 0:
            return [-b / (2 * a)]
        if isinstance(discriminant, complex) or discriminant < 0:
            root = cmath.sqrt(discriminant)
            return [(-b + root) / (2 * a), (-b - root) / (2 * a)]

        # Avoid subtracting nearly equal numbers for real roots.
        root = math.sqrt(discriminant)
        q = -(b + math.copysign(root, b)) / 2
        return [q / a, c / q]

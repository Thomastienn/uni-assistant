from sympy import CRootOf, Dummy, Expr, Poly, QQ, Symbol
from sympy.polys.polyerrors import CoercionFailed, PolynomialError

x = Symbol("x")

__all__ = ["Poly", "x"]


def reduce_root_expression(value):
    if not isinstance(value, Expr):
        return None
    roots = value.atoms(CRootOf)
    if len(roots) != 1:
        return None
    root = next(iter(roots))
    variable = Dummy("root")
    numerator, denominator = value.xreplace({root: variable}).as_numer_denom()
    try:
        numerator = Poly(numerator, variable, domain=QQ)
        denominator = Poly(denominator, variable, domain=QQ)
    except (CoercionFailed, PolynomialError):
        return None
    modulus = Poly(root.poly.as_expr().subs(root.poly.gen, variable), variable)
    # If p(root) = 0, polynomials differing by a multiple of p have the same value.
    reduced = (numerator.rem(modulus) * denominator.invert(modulus)).rem(modulus)
    return reduced.as_expr().xreplace({variable: root})

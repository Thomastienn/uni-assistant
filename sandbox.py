import sys
import math
from fractions import Fraction
# sys.path.append("D:\\cpFiles\\uni\\helper")

from linear_algebra.polynomial import *
from linear_algebra.matrix import *
from linear_algebra.customcomplex import *
from linear_algebra.linear_transformation import *


def func(vec):
    row1 = [vec.vR(0)+2*vec.vR(1)]
    row2 = [vec.vR(0) - vec.vR(1)]
    return Matrix(a=[row1, row2])


a = LinearTransformation(2, 2, func)
a.get_transform_mat().show()

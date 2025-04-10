from fractions import Fraction
import math


class Vector:
    def __init__(self, a=None, t=Fraction):
        self.t = t
        if a is None:
            self.get()
        else:
            self.a = a

    def get(self):
        self.a = list(map(self.t, input().split()))

    def __add__(self, oth):
        new = [self.a[i]+oth.a[i] for i in range(len(self.a))]
        return Vector(a=new, t=self.t)

    def __neg__(self):
        return Vector(a=[-self.a[i] for i in range(len(self.a))], t=self.t)

    def __sub__(self, o):
        return self+(-o)

    def __mul__(self, num):
        if isinstance(num, Vector):
            assert len(self.a) == len(num.a), "Not equal size"
            return sum(self.a[i]*num.a[i] for i in range(len(self.a)))
        new = [self.a[i]*num for i in range(len(self.a))]
        return Vector(a=new, t=self.t)

    def __rmul__(self, num):
        return self*num

    def __repr__(self):
        return " ".join(map(str, self.a))

    def smag(self):
        return sum(self.a[i]**2 for i in range(len(self.a)))

    def mag(self):
        return self.smag()**0.5

    def angle(self, other):
        return math.acos((self*other)/math.sqrt(self.smag()*other.smag()))

    def degAngle(self, other):
        return self.angle(other)*180/math.pi

    def piAngle(self, other):
        cof = self.angle(other)/math.pi
        return f"{cof:.2f}π"


class Vec3d:
    def __init__(self, x=None, y=None, z=None, t=Fraction):
        if (x == None):
            self.t = t
            self.get()
            return
        self.x = t(x)
        self.y = t(y)
        self.z = t(z)
        self.t = t

    def get(self):
        self.x, self.y, self.z = map(self.t, input().split())

    def __repr__(self):
        return f"<{self.x},{self.y},{self.z}>"

    def cross(self, other):
        new_x = self.y*other.z-self.z*other.y
        new_y = -(self.x*other.z-self.z*other.x)
        new_z = self.x*other.y-self.y*other.x
        return Vec3d(new_x, new_y, new_z, t=self.t)

    def dot(self, other):
        return self.x*other.x + self.y*other.y + self.z*other.z

    def __mul__(self, num):
        new_x = self.x*num
        new_y = self.y*num
        new_z = self.z*num
        return Vec3d(new_x, new_y, new_z, t=self.t)

    def __neg__(self):
        return Vec3d(-self.x, -self.y, -self.z, t=self.t)

    def __add__(self, other):
        new_x = self.x+other.x
        new_y = self.y+other.y
        new_z = self.z+other.z
        return Vec3d(new_x, new_y, new_z, t=self.t)

    def __sub__(self, other):
        return self+(-other)

    def __rmul__(self, num):
        new_x = self.x*num
        new_y = self.y*num
        new_z = self.z*num
        return Vec3d(new_x, new_y, new_z, t=self.t)

    def __truediv__(self, num):
        new_x = self.x*Fraction(1, num)
        new_y = self.y*Fraction(1, num)
        new_z = self.z*Fraction(1, num)
        return Vec3d(new_x, new_y, new_z, t=self.t)

    def smag(self):
        return self.x**2 + self.y**2 + self.z**2

    def mag(self):
        return self.smag() ** (0.5)


a = Vec3d(0, 4, 4, t=float)
b = Vec3d(-3, 5, 4, t=float)
c = Vec3d(5, -6, 0, t=float)
y = Vec3d(8, -5, 0, t=float)

print((c.dot(c) * a).dot(a))

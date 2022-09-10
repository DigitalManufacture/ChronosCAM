
__author__ = "Anthony Beaucamp"
__copyright__ = "Copyright (C) 2020 Kyoto University"
__license__ = "Public Domain"
__version__ = "1.0"

from numpy import *
from math import *

###############################################################################

def Normalize(vect):
    """!
    Normalize a vector
    @param vect: vector object
    @return: normalized vector
    """
    norm = linalg.norm(vect)
    if norm == 0:
        return vect
    return vect / norm

###############################################################################

def RotationABC(A, B, C):
    """!Compute Rotation matrix around X/Y/Z axes
    @param A: float
    @param B: float
    @param C: float
    """
    sn = sin(A); cn = cos(A)
    RotX = array([[ 1,  0,   0  ],
                  [ 0,  cn, -sn ],
                  [ 0,  sn,  cn ]])
    sn = sin(B); cn = cos(B)
    RotY = array([[ cn,  0,  sn ],
                  [ 0 ,  1,  0  ],
                  [-sn,  0,  cn ]])
    sn = sin(C); cn = cos(C)
    RotZ = array([[ cn, -sn,  0 ],
                  [ sn,  cn,  0 ],
                  [ 0 ,  0,   1 ]])
    return RotZ.dot(RotY.dot(RotX))
    
###############################################################################

def RotationAxis(axis, theta):
    """!
    Rotation matrix around specified axis (using Rodrigues Formula)
    @param axis: axis of rotation
    @param theta: angle of rotation
    @return: array (rotation matrix)
    """
    axis = axis / sqrt(dot(axis, axis))
    a = cos(theta / 2)
    b, c, d = -axis * sin(theta / 2)
    return array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                  [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                  [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])

###############################################################################

def CumDistance(points):
    """!
    Cumulative sum of the elements along a given axis.
    @param points:
    @return: cumulative sum
    """
    deltas = sqrt(diff(points[0])**2 + diff(points[1])**2 + diff(points[2])**2)
    return insert(cumsum(deltas), 0, 0)


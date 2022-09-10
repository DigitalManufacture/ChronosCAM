import PQP.PQP as ColDet
from OCC.Extend.DataExchange import write_stl_file
import numpy as np

def CreatePQPCollider(shape, linear_deflection=0.01):
    """!
    Generates a collider object for the supplied shape
    @param shape:
    @param world:
    @return Collider: return a PQP Model Object
    """
    # Write shape as Mesh file for collision detection
    write_stl_file(shape, "temp.stl", linear_deflection=linear_deflection)

    # Create Simulation World
    collider = ColDet.PQP_Model()

    ColDet.LoadModel(collider, "temp.stl")

    return collider


def CreatePQPRotationMatrix(quaternion=None):    
    # Rotation is 3X3 matrix
    vector = ColDet.DoubleVector(9)
    if quaternion is None:
        vector[0] = vector[4] = vector[8] = 1.
    else:
        mat = quaternion.GetMatrix()
        rotMatrix = np.array([[mat.Row(1).X(), mat.Row(1).Y(), mat.Row(1).Z()],
                              [mat.Row(2).X(), mat.Row(2).Y(), mat.Row(2).Z()],
                              [mat.Row(3).X(), mat.Row(3).Y(), mat.Row(3).Z()]])
        matrix = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        matrix = np.matmul(rotMatrix, matrix)
        vector[0] = matrix[0][0]
        vector[1] = matrix[0][1]
        vector[2] = matrix[0][2]
        vector[3] = matrix[1][0]
        vector[4] = matrix[1][1]
        vector[5] = matrix[1][2]
        vector[6] = matrix[2][0]
        vector[7] = matrix[2][1]
        vector[8] = matrix[2][2]
    return vector


def CreatePQPTranslationVector(offset=None):
    # Translation is 1X3 vector
    vector = ColDet.DoubleVector(3)
    if offset is not None:
        vector[0] = offset[0]
        vector[1] = offset[1]
        vector[2] = offset[2]
    return vector

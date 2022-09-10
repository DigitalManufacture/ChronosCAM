__author__ = "Anthony Beaucamp"
__copyright__ = "Copyright (C) 2020 Kyoto University"
__license__ = "Public Domain"
__version__ = "1.0"

import ode
from ode import collide
from OCC.MSH.Mesh import QuickTriangleMesh
from OCC.DYN.Context import DynamicShape, DynamicSimulationContext
from OCC.Tools.Viewer import *
from OCC.Extend.DataExchange import write_stl_file


def CreateCollider(shape, world):
    """!
    Generates a collider object for the supplied shape
    @param shape:
    @param world:
    @return Collider: return a DynamicShape ODE collider
    """
    # if type(shape) is ctm.Mesh:
    #     ### Compute ODE collider from CTM Mesh
    #     faces = []
    #     for i in range(ctm.NumTriangles(shape)):
    #         i1 = ctm.TriangleNode1(shape,i)
    #         i2 = ctm.TriangleNode2(shape,i)
    #         i3 = ctm.TriangleNode3(shape,i)
    #         faces.append([i1,i2,i3])
    #     vertices = []
    #     for i in range(ctm.NumVertices(shape)):
    #         x = ctm.XCoordinateOfNode(shape,i)
    #         y = ctm.YCoordinateOfNode(shape,i)
    #         z = ctm.ZCoordinateOfNode(shape,i)
    #         vertices.append([x,y,z])
    # else:
    ### Compute ODE collider from TopoDS Shape
    mesh = QuickTriangleMesh()
    mesh.set_shape(shape)

    # define precision for the mesh
    quality_factor = 1.0
    mesh.set_precision(mesh.get_precision() / quality_factor)

    # then compute the mesh
    mesh.compute()

    # Extract vertices/faces
    vertices = mesh.get_vertices()
    f = mesh.get_faces()
    faces = [];
    i = 0
    while i < len(f):
        faces.append([i, i + 1, i + 2])
        i = i + 3

    # Create the ODE trimesh data
    td = ode.TriMeshData()
    td.build(vertices, faces)

    # Save as ODE collider
    collider = DynamicShape(world)
    collider.geometry = ode.GeomTriMesh(td, world.get_collision_space())
    collider.geometry.setBody(collider)
    collider.enable_collision_detection()
    return collider


def UpdateCollider(collider, trsf):
    """!
    Applies a tranform to the Collider object
    @param collider: collider object to apply a transform to
    @param trsf: transforms (postion & rotation)
    """

    # Copy transform to ODE
    collider.setPosition([trsf.Value(1, 4), trsf.Value(2, 4), trsf.Value(3, 4)])
    collider.setRotation([trsf.Value(1, 1), trsf.Value(1, 2), trsf.Value(1, 3),
                          trsf.Value(2, 1), trsf.Value(2, 2), trsf.Value(2, 3),
                          trsf.Value(3, 1), trsf.Value(3, 2), trsf.Value(3, 3)])


def CreateAdvanceCollider():
    """!
    Creates an instance of the CollisonDetector class and returns a reference to the instance.
    @return: Instance of CollisionDetector object.
    """
    return CollisionDetector('ColDet-6.7', 'Zeeko_Framework         0        0      0    0 eabb4636fd6203dc7f41')


def UpdateAdvanceCollider(collider, toolId, ctr, dir, angle, offsetVector):
    """!
    Updates the rotation and translation of the the passed collider object.
    @param collider: DynamicShape object
    @param toolId: tool identifier
    @param ctr: center of rotation vector (xRot, yRot, zRot)
    @param dir: direction vector (xDir, yDir, zDir)
    @param angle: angle of rotation
    @param offsetVector: offset vector (xPos, yPos, zPos)
    """
    collider.Reset()
    collider.RotateEntity(toolId, angle, ctr.X(), ctr.Y(), ctr.Z(), dir.X(), dir.Y(), dir.Z())
    collider.TranslateEntity(toolId, offsetVector.X(), offsetVector.Y(), offsetVector.Z())


def LoadEntity(collider, shapeOrFilepath):
    """!
    Instantiates a new Collision Entity from a Shape or a File object
    @param collider: DynamicShape object
    @param shapeOrFilepath: shape object or a file location on disk
    @return: int identifying the instantiated object
    """
    if type(shapeOrFilepath) == str:
        shape = occimport(filepath)
    elif type(shapeOrFilepath) == list:
        shape = shapeOrFilepath
    else:
        shape = [shapeOrFilepath]
    occexport('temp.brep', shape)
    newId = collider.LoadEntity("temp.brep")
    os.remove('temp.brep')
    return newId

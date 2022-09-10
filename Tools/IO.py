
__author__ = "Anthony Beaucamp"
__copyright__ = "Copyright (C) 2020 Kyoto University"
__license__ = "Public Domain"
__version__ = "1.0"

from numpy import *
from stl import *
import sys, os, os.path
from OCC.Extend.DataExchange import write_iges_file, write_stl_file, write_step_file, read_step_file
from OCC.Core import TopoDS, BRep, BRepTools, StlAPI, IGESControl
from OCC.Core.BRep import *

def occimport(filepath):
    """!
    Reads CAD data from different file types.
    Supported file formats: .iges, .igs, .step, .stp, .stl, .bRep
    @param filepath: file path on disk
    @return: shape object
    """

    # Use relevant file importer
    extension = os.path.basename(filepath).split(".").pop().lower()
    if extension == "iges" or extension == "igs":
        # Read IGS data
        i = IGESControl.IGESControl_Controller(); i.Init()
        iges_reader = IGESControl.IGESControl_Reader()
        iges_reader.ReadFile(str(filepath))
        iges_reader.TransferRoots()
        # nShapes = iges_reader.NbShapes()
        shape = iges_reader.OneShape()
        shapes = [shape]

    elif extension == "step" or extension == "stp":
        # Read STP data
        shapes = [read_step_file(str(filepath))]

    elif extension == "stl":
        from OCC.Core import TopoDS, StlAPI
        shape = TopoDS.TopoDS_Shape()
        stl_reader = StlAPI.StlAPI_Reader()
        stl_reader.Read(shape, str(filepath))
        shapes = [shape]

    elif extension == "brep":
        from OCC.Core import TopoDS
        # Read brep data
        compShape = TopoDS.TopoDS_Compound()
        builder = BRep.BRep_Builder()
        BRepTools.breptools().Read(compShape, str(filepath), builder)
        shapeIterator = TopoDS.TopoDS_Iterator(compShape)
        shapes = []
        while shapeIterator.More():
            shapes.append(shapeIterator.Value())

    return shapes

def occexport(filepath, shapes):
    """!
    Exports a shape for different file formats and serializes the on disk at the specified file path.
    Supported file formats: .igs, .iges, .step, .stp, .bRep
    @param filepath: file location on disk
    @param shapes: shapes object
    """

    # Create a TopoDS Compund Shape containing all TopoDS Shapes
    compShape = TopoDS.TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compShape)

    # Add shape to Compound
    for shape in shapes:
        builder.Add(compShape, shape)

    # Output shape file format
    outFileExtension = os.path.splitext(os.path.basename(filepath))[1].lower()
    if outFileExtension in [".igs", ".iges"]:
        write_iges_file(compShape, str(filepath))

    elif outFileExtension in [".stp", ".step"]:
        write_step_file(compShape, str(filepath))

    elif outFileExtension == ".brep":
        BRepTools.breptools_Write(compShape, str(filepath))
        
def stlexport(filepath, vertices, triangles):
    stlMesh = Mesh(zeros(len(triangles), dtype=Mesh.dtype))
    for i in range(len(triangles)):
        for j in range(3):
            vIndex = triangles[i][j]
            stlMesh.vectors[i][j][0] = vertices[0][vIndex]
            stlMesh.vectors[i][j][1] = vertices[1][vIndex]
            stlMesh.vectors[i][j][2] = vertices[2][vIndex]
    stlMesh.save(filepath)    

def ctmimport(filepath):
    """!
    Import mesh data from a file object
    Supported file formats: .3ds, .stl, .ctm, .lwo, .dae , .obj, .off
    @param filepath: file location on disk
    @return: mesh object if successful else None
    """
    # Use relevant file importer
    extension = os.path.basename(filepath).split(".").pop().lower()
    if extension == "stl":
        aMesh = ctm.Mesh()
        # Import binary/ascii stl mesh
        basename = os.path.splitext(os.path.basename(str(filepath)))[0]
        dirname = os.path.dirname(str(filepath))
        binaryFilepath = os.path.join(dirname, basename + '_binaryMesh.stl')
        if ctm.ConvertAsciiStlToBinary(str(filepath), binaryFilepath):
            if os.path.exists(binaryFilepath):
                ctm.OCCCTM_Import(binaryFilepath, aMesh)
            try:
                os.remove(binaryFilepath)
            except:
                pass
        if ctm.NumVertices(aMesh) == 0:
            # Read the mesh
            aMesh = ctm.Mesh()
            # OCCCTM_Import(str(self.filepath), ctmMesh)
            ctm.OCCCTM_Import(str(filepath), aMesh)
        return aMesh
    elif extension in ["ctm", "3ds", "lwo", "dae", "obj", "off"]:
        # Read Meshes
        aMesh = ctm.Mesh()
        ctm.OCCCTM_Import(str(filepath), aMesh)
        return aMesh
    else:
        print("Mesh file invalid or not supported.")
        return None

def ctmexport(aMesh, filepath):
    """!
    Serialize mesh object to a file on the disk
    @param aMesh: mesh object
    @param filepath: file save location on disk
    @return: bool fail or success
    """
    try:
        ctm.OCCCTM_Export(aMesh, str(filepath))
        return True
    except:
        print("Failed to write the mesh.")
        return False
        

###############################################################################
'''    
from OCC.MeshToMeshVS import *
    
def occmeshVS(polygons):
    # Init data
    faces = []
    facesTri = []
    vertices = []
    normals = []
    
    # Get Polygon Mesh data and convert into triangular mesh
    for polygon in polygons:
        indices = []
        for v in polygon.vertices:
            pos = [v.pos.x, v.pos.y, v.pos.z]
            if not pos in vertices:
                vertices.append(pos)
            index = vertices.index(pos)
            indices.append(index)
        faces.append(indices)
        if len(indices) == 3:
            facesTri.append(indices)
        elif len(indices) == 4:
            facesTri.append([indices[0],indices[1],indices[2]])
            facesTri.append([indices[2],indices[3],indices[0]])
        elif len(indices) > 4:
            start = indices[0]
            facesTri.append([indices[0],indices[1],indices[2]])
            indices.remove(indices[0])
            indices.remove(indices[0])
            triList = []
            for index in range(len(indices)-1):
                triList.append([indices[index],indices[index+1]])
            for i, item in enumerate(triList):
                if i%2 == 0:
                    triList[i].append(start)
                else:
                    triList[i].insert(0,start)
                facesTri.append(triList[i])
        n = polygon.plane.normal
        normals.append([n.x, n.y, n.z])
    
    # Convert Mesh Data into Vector format required by Mesh_MeshVSLink
    meshFaces = IntVector(3*(len(facesTri)))
    meshVertices = DoubleVector(3*len(vertices))
    for i, face in enumerate(facesTri):
        meshFaces[i*3] = face[0]
        meshFaces[(i*3 + 1)] = face[1]
        meshFaces[(i*3 + 2)] = face[2]
    for i, vertex in enumerate(vertices):
        meshVertices[i*3] = vertex[0]
        meshVertices[i*3 + 1] = vertex[1]
        meshVertices[i*3 + 2] = vertex[2]
        
    # Data Source (MeshVS)
    return Mesh_MeshVSLink(meshFaces, meshVertices)
'''


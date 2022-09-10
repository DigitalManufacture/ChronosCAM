__author__ = "Anthony Beaucamp"
__copyright__ = "Copyright (C) 2020 Kyoto University"
__license__ = "Public Domain"
__version__ = "1.0"

import sys, os, os.path
from math import *
from numpy import *
from numpy.linalg import *
from matplotlib.path import Path
from copy import deepcopy

from OCC.Core import TopoDS
from OCC.Core.TopTools import *
from OCC.Core.gp import *
from OCC.Core.Bnd import *
from OCC.Core.BRep import *
from OCC.Core.BRepAlgo import *
from OCC.Core.BRepAlgoAPI import *
from OCC.Core.BRepAdaptor import *
from OCC.Core.BRepBndLib import *
from OCC.Core.BRepBuilderAPI import *
from OCC.Core.BRepExtrema import *
from OCC.Core.BRepFeat import *
from OCC.Core.BRepFill import *
from OCC.Core.BRepGProp import *
from OCC.Core.BRepLProp import *
from OCC.Core.BRepPrimAPI import *
from OCC.Core.BRepTools import *
from OCC.Core.GCPnts import *
from OCC.Core.GeomAPI import *
from OCC.Core.GProp import *
from OCC.Core.IntCurvesFace import *
from OCC.Core.ShapeAnalysis import *
from OCC.Core.ShapeFix import *
from OCC.Core.ShapeExtend import *
from OCC.Core.TopAbs import *
from OCC.Core.TopLoc import *
from OCC.Core.TopExp import *
from OCC.Core.Geom import *
from OCC.Core.GeomAbs import *
from OCC.Core.GeomAPI import *
from OCC.Core.GeomAdaptor import *
from OCC.Core.GeomLib import *
from OCC.Core.GeomLProp import *
from OCC.Core.GeomProjLib import *
from OCC.Core.LocOpe import *
from OCC.Core.ShapeFix import *
from OCC.Core.ShapeExtend import *
from OCC.Core.ShapeAnalysis import *
from OCC.Core.TopAbs import *
from OCC.Core.TopLoc import *

from OCC.Utils.Topology import *

from OCC.Core.Graphic3d import *
from OCC.Core.Quantity import *
from OCC.Core.Aspect import *
from OCC.Core.MeshVS import *
from OCC.Core.AIS import *

from OCC.Core import VERSION
import OCC.Core.TopTools as TopTools
import OCC.Core.TopoDS as TopoDS	


###############################################################################

def face2shell(faces):
    """!Compute shell from selected faces
    @return: TopoDS.TopoDS_Shell shell
    """
    builder = BRep_Builder()
    shell = TopoDS.TopoDS_Shell()
    builder.MakeShell(shell)

    # Compute shell from selected faces
    for face in faces:
        builder.Add(shell, face)
            
    return shell


def Normalize(vect):
    norm = linalg.norm(vect)
    if norm == 0:
        return vect
    return vect / norm
  
    
def section2points(section, shape, direction, sampling, pointMode=0, strangeCAD=False, hashKey=None, self=None, trackSpacing=None,
            silent=False, autoFixNormals=True, selectedArea=0., extraShapes=[], arcInterp=False):

    # Half the point spacing in case of G02/G03 interpolation for CNC output
    if arcInterp is True:
        sampling /= 2.
        
    section.Approximation(True)
    section.ComputePCurveOn1(True)
    sectionFailed = False
    try:
        section.Build()
    except:
        sectionFailed = True
    if sectionFailed or not section.IsDone():
        # Return Empty Data
        points = array([[], [], []])
        normals = array([[], [], []])
        tangents = array([[], [], []])
        fuv = array([[], [], []])
        curvature = array([[], []])
        return points, normals, tangents, fuv

    # List of wires from Section Curve
    newWires = WiresFromSectioning(section)

    ############################
    # Perform Sampling of Wires
    points = array([[], [], []])
    normals = array([[], [], []])
    tangents = array([[], [], []])
    fuv = array([[], [], []])
    curvature = array([[], []])
    edgeMarkers = []

    # Initialize OCC vars
    evalPnt = gp_Pnt()
    evalPnt2d = gp_Pnt2d()
    evalTan = gp_Vec()
    
    # Autofix flags
    autoFixNormals = True

    # Loop through wires
    for wire in newWires:

        # Dictionary for storing crvPoints, crvNormals, crvTangents for correcting their order later
        edgeIndex = 0
        edgePointDict = {}
        edgeNormalDict = {}
        edgeTangentDict = {}
        edgeFuvDict = {}
        edgeCurvatureDict = {}

        # Explore wire (to get edges)
        exp = BRepTools_WireExplorer(TopoDS.topods_Wire(wire))
        while exp.More():
            # Init edge and face
            edge = exp.Current()
            face = TopoDS.topods_Face(TopoDS.TopoDS_Shape())
            # Hack for HasAncestorFaceOn1 error (look-up original edge handle)
            if section.HasAncestorFaceOn1(edge, face) == 0:
                curve = BRepAdaptor_Curve(edge)
                startPoint = curve.Value(curve.FirstParameter())
                endPoint = curve.Value(curve.LastParameter())
                for anedge in Topo(section.Shape()).edges():
                    acurve = BRepAdaptor_Curve(anedge)
                    aStartPoint = acurve.Value(acurve.FirstParameter())
                    anEndPoint = acurve.Value(acurve.LastParameter())
                    if startPoint == aStartPoint and endPoint == anEndPoint:
                        edge = anedge
                        break

            # Look-up parent Face
            aFaceIndex = -1
            if section.HasAncestorFaceOn1(edge, face):
                # Get face index in the workpiece faces for fuv if above fails
                topExp = TopExp_Explorer()
                topExp.Init(TopoDS.TopoDS_Shape(shape), TopAbs_FACE)
                shellFaceIndex = 0
                while topExp.More():
                    aShellFace = TopoDS.topods_Face(topExp.Current())
                    if face == aShellFace:
                        aFaceIndex = shellFaceIndex
                        break
                    topExp.Next()
                    shellFaceIndex += 1

                if aFaceIndex == -1:
                    try:
                        for anObject in session.objects:
                            for ijk, aFace in enumerate(anObject.faces):
                                if aFace == face:
                                    aFaceIndex = ijk
                                    break
                            if aFaceIndex != -1:
                                break
                    except:
                        for aShape in extraShapes:
                            ijk = 0
                            for aExtraFace in Topo(aShape).faces():
                                if aExtraFace == face:
                                    aFaceIndex = ijk
                                    break
                                ijk += 1
                            if aFaceIndex != -1:
                                break

                # --- Get Geometric Surface ---
                surfHandle = BRep_Tool_Surface(face)
                surf = BRepLProp_SLProps(BRepAdaptor_Surface(face), 1, 0.000001)

                # Extract UV Curve
                curve2D, uMin2D, uMax2D = BRep_Tool_CurveOnSurface(edge, face)
                curve2D = curve2D

                # Extract 3D Curve
                curve3DHandle, uMin, uMax = BRep_Tool_Curve(edge)
                curve3D = curve3DHandle

                # Get Linear Properties
                props = GProp_GProps()
                brepgprop_LinearProperties(edge, props)
                length = props.Mass()

                # Initialize Numpy Curves
                gac = GeomAdaptor_Curve(curve3DHandle)
                gcua = GCPnts_UniformAbscissa(gac, sampling, uMin, uMax)
                nPoints = gcua.NbPoints()
                if gcua.IsDone() and gcua.NbPoints() >= 2:
                    nPoints = gcua.NbPoints()
                    gcua = GCPnts_QuasiUniformAbscissa(gac, gcua.NbPoints(), uMin, uMax)
                    if gcua.IsDone():
                        nPoints = gcua.NbPoints()

                # --- Compute Curve Points ---
                cLen = max(nPoints, 2)
                uRef = linspace(uMin, uMax, cLen)
                uRef2D = linspace(uMin2D, uMax2D, cLen)

                crvPoints = zeros((3, cLen))
                crvNormals = zeros((3, cLen))
                crvTangents = zeros((3, cLen))
                crvFuv = zeros((3, cLen))
                crvCurvature = zeros((2, cLen))
                crvU = zeros(cLen)
                crvV = zeros(cLen)

                # --- Compute Curve Points ---
                for u in range(cLen):
                    # Eval Point and Tangent
                    if gcua.IsDone() and gcua.NbPoints() >= 2:
                        curve3D.D1(gcua.Parameter(u + 1), evalPnt, evalTan)
                    else:
                        curve3D.D1(uRef[u], evalPnt, evalTan)

                    # Old algorithm (slow)
                    # projection = GeomAPI_ProjectPointOnSurf(evalPnt, surfHandle)
                    # evalU, evalV = projection.LowerDistanceParameters()
                    # surf.SetParameters(evalU, evalV)

                    # Find U/V Coordinates on Surface
                    if gcua.IsDone() and gcua.NbPoints() >= 2:
                        curve2D.D0(gcua.Parameter(u + 1), evalPnt2d)
                    else:
                        curve2D.D0(uRef2D[u], evalPnt2d)
                    crvU[u] = evalPnt2d.X()
                    crvV[u] = evalPnt2d.Y()

                    # Eval Point (again just to double check) and Normal
                    try:
                        # from OCC.GeomLProp import *
                        # prop = GeomLProp_SLProps(surfHandle, crvU[u], crvV[u], 1, 0.01 )
                        # evalNor = prop.Normal()
                        surf.SetParameters(crvU[u], crvV[u])
                        evalNor = surf.Normal()
                        evalPnt = surf.Value()
                    except:
                        point3d = gp_Pnt()
                        uTan = gp_Vec()
                        vTan = gp_Vec()
                        surfHandle.D1(crvU[u], crvV[u], point3d, uTan, vTan)
                        if uTan.Magnitude() < 1e-3 or vTan.Magnitude() < 1e-3:
                            if point3d.SquareDistance(gp_Pnt(0., 0., 0.)) < 1e-3:
                                evalNor = gp_Dir(gp_Vec(0, 0, 1))
                        else:
                            uTan.Normalize()
                            vTan.Normalize()
                            evalNor = gp_Dir(uTan.Crossed(vTan))
                        surfHandle.D0(crvU[u], crvV[u], evalPnt)
                    if face.Orientation() == TopAbs_REVERSED or evalNor.Z() < 0.:
                        evalNor.Reverse()

                    # Eval Curvature
                    if gcua.IsDone() and gcua.NbPoints() >= 2:
                        glpClp = GeomLProp_CLProps(curve3DHandle, 2, 0.0001)
                        glpClp.SetParameter(gcua.Parameter(u + 1))
                    else:
                        glpClp = GeomLProp_CLProps(curve3DHandle, 2, 0.0001)
                        glpClp.SetParameter(uRef[u])
                    try:
                        aCurvature = glpClp.Curvature()
                    except:
                        aCurvature = 0.0

                    # Save in NumPy format
                    crvPoints[0, u] = evalPnt.X()
                    crvPoints[1, u] = evalPnt.Y()
                    crvPoints[2, u] = evalPnt.Z()
                    magnitude = sqrt(evalNor.X() ** 2 + evalNor.Y() ** 2 + evalNor.Z() ** 2)
                    crvNormals[0, u] = evalNor.X() / magnitude
                    crvNormals[1, u] = evalNor.Y() / magnitude
                    crvNormals[2, u] = evalNor.Z() / magnitude
                    magnitude = sqrt(evalTan.X() ** 2 + evalTan.Y() ** 2 + evalTan.Z() ** 2)
                    crvTangents[0, u] = evalTan.X() / magnitude
                    crvTangents[1, u] = evalTan.Y() / magnitude
                    crvTangents[2, u] = evalTan.Z() / magnitude
                    crvFuv[0, u] = aFaceIndex
                    crvFuv[1, u] = crvU[u]
                    crvFuv[2, u] = crvV[u]
                    crvCurvature[0, u] = aCurvature
                    crvCurvature[1, u] = aCurvature

                # --- Check Tangent Direction ---
                crvDir = crvPoints[:, 1:cLen] - crvPoints[:, 0:cLen - 1]
                crvDir = append(crvDir, crvDir[:, cLen - 2:cLen - 1], axis=1)
                for n in range(cLen):
                    if dot(crvDir[:, n], crvTangents[:, n]) < 0:
                        crvTangents[:, n] = -crvTangents[:, n]

                # Create dictionary for points, normals, tangents and edges
                edgePointDict[edgeIndex] = crvPoints
                edgeNormalDict[edgeIndex] = crvNormals
                edgeTangentDict[edgeIndex] = crvTangents
                edgeFuvDict[edgeIndex] = crvFuv
                edgeCurvatureDict[edgeIndex] = crvCurvature
                edgeIndex = edgeIndex + 1

                # Save edge in edgeMarkers (for display)
                edgeMarkers.append(edge)

            exp.Next()

        # The original edge/curve order given by BRep_WireExplorer
        origCrvOrder = []
        for i in range(edgeIndex):
            origCrvOrder.append(i)

        # Check for duplicate edges and remove them if found
        copyEdgePointDict = {}
        duplicateList = []
        for i in range(edgeIndex):
            found = False
            if len(copyEdgePointDict) == 0:
                copyEdgePointDict[i] = edgePointDict[i]
            else:
                for j in range(len(copyEdgePointDict)):
                    if j in duplicateList:
                        continue
                    if edgePointDict[i].shape[1] == copyEdgePointDict[j].shape[1]:
                        if array_equal(edgePointDict[i], copyEdgePointDict[j]):
                            found = True
                            duplicateList.append(i)
                            break
                        elif array_equal(edgePointDict[i], fliplr(copyEdgePointDict[j])):
                            found = True
                            duplicateList.append(i)
                            break
                if found is False:
                    copyEdgePointDict[i] = edgePointDict[i]
        # End of Duplicate edges removal

        # Check edges for order, flip them if required
        crvOrder = []
        tol = 0.0001
        while len(crvOrder) < (edgeIndex - len(duplicateList)):
            if len(crvOrder) == 0:
                crvOrder.append(0)
                if edgeIndex > 1:
                    if sum((edgePointDict[0][:, 0] - edgePointDict[1][:, 0]) ** 2) < sum(
                            (edgePointDict[0][:, -1] - edgePointDict[1][:, 0]) ** 2) \
                            and sum((edgePointDict[0][:, 0] - edgePointDict[1][:, -1]) ** 2) < sum(
                        (edgePointDict[0][:, -1] - edgePointDict[1][:, -1]) ** 2):
                        edgePointDict[0] = fliplr(edgePointDict[0])
                        edgeNormalDict[0] = fliplr(edgeNormalDict[0])
                        edgeTangentDict[0] = -fliplr(edgeTangentDict[0])
                        edgeFuvDict[0] = fliplr(edgeFuvDict[0])
                        edgeCurvatureDict[0] = fliplr(edgeCurvatureDict[0])
                currentCurve = edgePointDict[0]
                currCurveLen = edgePointDict[0].shape[1]
            else:
                checkLength = len(crvOrder)
                for i in range(edgeIndex):
                    if i in crvOrder or i in duplicateList:
                        continue
                    cLen = edgePointDict[i].shape[1]
                    if (abs(edgePointDict[i][0, 0] - currentCurve[0, currCurveLen - 1]) < tol) and \
                            (abs(edgePointDict[i][1, 0] - currentCurve[1, currCurveLen - 1]) < tol) and \
                            (abs(edgePointDict[i][2, 0] - currentCurve[2, currCurveLen - 1]) < tol):
                        crvOrder.append(i)
                        currentCurve = edgePointDict[i]
                        currCurveLen = edgePointDict[i].shape[1]
                        break
                    elif (abs(edgePointDict[i][0, cLen - 1] - currentCurve[0, currCurveLen - 1]) < tol) and \
                            (abs(edgePointDict[i][1, cLen - 1] - currentCurve[1, currCurveLen - 1]) < tol) and \
                            (abs(edgePointDict[i][2, cLen - 1] - currentCurve[2, currCurveLen - 1]) < tol):
                        crvOrder.append(i)
                        edgePointDict[i] = fliplr(edgePointDict[i])
                        edgeNormalDict[i] = fliplr(edgeNormalDict[i])
                        edgeTangentDict[i] = -fliplr(edgeTangentDict[i])
                        edgeFuvDict[i] = fliplr(edgeFuvDict[i])
                        edgeCurvatureDict[i] = fliplr(edgeCurvatureDict[i])
                        currentCurve = edgePointDict[i]
                        currCurveLen = edgePointDict[i].shape[1]
                        break
                if checkLength == len(crvOrder):
                    tol = tol * 2
                else:
                    tol = 0.0001

        # Check CCW orientation (for closed loops)
        if wire.Closed():
            # Find the longest edge (edge with maximum points)
            edgeLengthList = []
            for m in range(edgeIndex):
                edgeLengthList.append(edgePointDict[m].shape[1])
            if not edgeLengthList:
                # return edgePointDict, edgeNormalDict, edgeTangentDict, edgeFuvDict, edgeMarkers, crvOrder
                # Add the points to the final points array in correct order
                for edgePntInd in range(len(list(edgePointDict.keys()))):
                    try:
                        points = append(points, edgePointDict[edgePntInd], axis=1)
                        normals = append(normals, edgeNormalDict[edgePntInd], axis=1)
                        tangents = append(tangents, edgeTangentDict[edgePntInd], axis=1)
                        fuv = append(fuv, edgeFuvDict[edgePntInd], axis=1)
                        curvature = append(curvature, edgeCurvatureDict[edgePntInd], axis=1)
                    except:
                        pass
                # return edgePointDict, edgeNormalDict, edgeTangentDict, edgeFuvDict, edgeMarkers
                return points, normals, tangents, fuv, markers
            maxValue = max(edgeLengthList)
            maxIndex = edgeLengthList.index(maxValue)

            # Find mid-point of longest edge
            copiedEdgePointDict = deepcopy(edgePointDict)
            copiedEdgeTangentDict = deepcopy(edgeTangentDict)
            midIndexInCrvPoints = maxValue // 2
            crvPointsLongestEdge = deepcopy(copiedEdgePointDict[maxIndex])
            midPointOnCrvPoints = crvPointsLongestEdge[:, midIndexInCrvPoints]

            # Compute cross product of direction and tangent, then project from mid-point
            crvTangentLongestEdge = deepcopy(copiedEdgeTangentDict[maxIndex])
            a = array([direction.X(), direction.Y(), direction.Z()])
            b = crvTangentLongestEdge[:, midIndexInCrvPoints]
            crossProduct = cross(a, b)
            # InPointToTest = midPointOnCrvPoints + 0.01*crossProduct
            InPointToTest = midPointOnCrvPoints + 0.5 * crossProduct

            # Find Rotation Matrix for rotating plane to align it with xy Plane
            tempPoints = array([[], [], []]);
            if a[0] == 0.00 and a[1] == 0.00 and a[2] != 0:
                # Store all the points in tempPoints
                for correctIndex in crvOrder:
                    tempPoints = append(tempPoints, edgePointDict[correctIndex], axis=1)
                # Delete the Z-coordinate to get 2-d planar points
                temp2dPoints = delete(tempPoints, (2), axis=0)

                # Set RotMatrix as Identity
                RotMatrix = array([[1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]])
            else:
                # Unit Vector in Z-dir
                b = array([0, 0, 1])
                # Get rotation axis
                axis = cross(b, Normalize(a))
                # Get rotation angle
                theta = acos((dot(a, b)) / (linalg.norm(a) * linalg.norm(b)))
                # Get rotation matrix
                RotMatrix = RotationAxis(axis, theta)

                # Get rotated points on the new plane parallel to x-y plane
                for i, correctIndex in enumerate(crvOrder):
                    correctIndex = abs(correctIndex)
                    anArray = array([[], [], []]);
                    for j in range(edgePointDict[correctIndex].shape[1]):
                        anArray = append(anArray,
                                         array([[RotMatrix.dot(copiedEdgePointDict[correctIndex][:, j])[0]],
                                                [RotMatrix.dot(copiedEdgePointDict[correctIndex][:, j])[1]],
                                                [RotMatrix.dot(copiedEdgePointDict[correctIndex][:, j])[2]]]),
                                         axis=1)
                    # Store points in tempPoints for Path method of matplotlib to check whether the InPointToTest lies inside or outside the loop
                    tempPoints = append(tempPoints, anArray, axis=1)

                # Remove the Z-coordinate from them as they are planar points
                temp2dPoints = delete(tempPoints, (2), axis=0)

                # Rotate the InPointToTest on the new plane
                InPointToTest = RotMatrix.dot(InPointToTest)

            # Check if point lies inside or outside the path
            InPointToTest = array([[InPointToTest[0]], [InPointToTest[1]]])
            p = Path(temp2dPoints.transpose())
            if p.contains_point(InPointToTest) != 1:
                # Change orientation (Flip points, normals and tangents)
                crvOrder = crvOrder[::-1]
                for index in crvOrder:
                    edgePointDict[index] = fliplr(edgePointDict[index])
                    edgeNormalDict[index] = fliplr(edgeNormalDict[index])
                    edgeTangentDict[index] = -fliplr(edgeTangentDict[index])
                    edgeFuvDict[index] = fliplr(edgeFuvDict[index])
                    edgeCurvatureDict[index] = fliplr(edgeCurvatureDict[index])
                # Flip the tempPoints and 2d points
                tempPoints = fliplr(tempPoints)
                temp2dPoints = fliplr(temp2dPoints)

        # Find the correct starting point and the edge containing it if wire is closed
        if wire.Closed():
            intersectPointsValues = {}
            normalVec = array([direction.X(), direction.Y(), direction.Z()])

            # Find centre point of bounding box of curve
            xMin = temp2dPoints.transpose()[argmin(temp2dPoints.transpose()[:, 0])][0]
            xMax = temp2dPoints.transpose()[argmax(temp2dPoints.transpose()[:, 0])][0]
            yMin = temp2dPoints.transpose()[argmin(temp2dPoints.transpose()[:, 1])][1]
            yMax = temp2dPoints.transpose()[argmax(temp2dPoints.transpose()[:, 1])][1]
            xCentre = (xMin + xMax) * 0.5
            yCentre = (yMin + yMax) * 0.5

            # If normal already in Z-direction
            if normalVec[0] == 0.00 and normalVec[1] == 0.00 and normalVec[2] != 0:
                # Given a Separate Value
                thetaAngle = 9999
                # Find a 2d point at unit distance from centre point in X direction as decided
                x2 = xCentre + cos(0)
                y2 = yCentre + sin(0)

            else:
                # Project normal vector on X-Y plane if not in Z direction
                nVec = array([0, 0, 1])
                dotProd = dot(normalVec, nVec)
                proj2dVec = array(
                    [(direction.X() - (dotProd * nVec[0])),
                     (direction.Y() - (dotProd * nVec[1]))])  # ignore z value
                xVec = array([1, 0])  # Ignore z, as we take 2d plane
                dotProduct = xVec[0] * proj2dVec[0] + xVec[1] * proj2dVec[1]  # dot product
                det = xVec[0] * proj2dVec[1] - xVec[1] * proj2dVec[0]  # determinant
                thetaAngle = atan2(det, dotProduct)
                # Find a 2d point at unit distance from centre point in thetaAngle direction if thetaAngle not infinite(normalVec already in z-dir case)
                x2 = xCentre + cos(thetaAngle)
                y2 = yCentre + sin(thetaAngle)

            # Rotate back thetaAngle direction to the original plane
            inverseRotMatrix = inv(RotMatrix)
            if thetaAngle != 9999:
                dirVector2d = array([[cos(thetaAngle)], [sin(thetaAngle)], [0]])
            else:
                dirVector2d = array([[cos(0)], [sin(0)], [0]])
            dirVector3d = Normalize(inverseRotMatrix.dot(dirVector2d))

            # Rotate centre point back to original plane and make an edge in the thetaAngle dir
            centrePoint3d = inverseRotMatrix.dot(array([[xCentre], [yCentre], [tempPoints[2, 0]]]))
            startPointOfEdge = centrePoint3d - 10000 * dirVector3d
            endPointOfEdge = centrePoint3d + 10000 * dirVector3d
            edge1 = BRepBuilderAPI_MakeEdge(
                gp_Pnt(startPointOfEdge[0, 0], startPointOfEdge[1, 0], startPointOfEdge[2, 0]),
                gp_Pnt(endPointOfEdge[0, 0], endPointOfEdge[1, 0], endPointOfEdge[2, 0])).Edge()

            # Find the edge1 intersection with edges on the current wire
            allIntersectionPoints = []
            allIntersectionPoints2d = []
            intersection2dPoints = {}
            for order in crvOrder:
                edge2 = edgeMarkers[order]
                dss = BRepExtrema_DistShapeShape(edge1, edge2)
                if dss.Value() < 0.0001:
                    intersectPoints = []
                    rotated2dPoint = []
                    # Store the intersection points in intersectPointsValues and intersection2dPoints dictionary
                    for i in range(1, dss.NbSolution() + 1):
                        containsPoint = False
                        # Check if the point already exists in the list
                        for apoint in allIntersectionPoints:
                            if apoint == dss.PointOnShape2(i):
                                containsPoint = True
                        if not containsPoint:
                            intersectPoints.append(dss.PointOnShape2(i))
                            allIntersectionPoints.append(dss.PointOnShape2(i))
                            # Rotate the point to the rotated X-Y plane
                            rotatedPoint = RotMatrix.dot(
                                array([[dss.PointOnShape2(i).X()], [dss.PointOnShape2(i).Y()],
                                       [dss.PointOnShape2(i).Z()]]))
                            rotated2dPoint.append(rotatedPoint[0:2, :])
                            allIntersectionPoints2d.append(rotatedPoint[0:2, :])
                    if intersectPoints:
                        intersectPointsValues[order] = intersectPoints
                        intersection2dPoints[order] = rotated2dPoint

            # Find the correct starting point among the intersection points if intersection happens
            if intersectPointsValues:
                xOrYList = []
                if (-pi) / 4 <= thetaAngle and thetaAngle <= pi / 4:
                    # Correct 2d point will be one with maximum X-Value
                    for points2d in allIntersectionPoints2d:
                        xOrYList.append(points2d[0, 0])
                    correct2dPointArray = allIntersectionPoints2d[xOrYList.index(max(xOrYList))]
                elif pi / 4 < thetaAngle and thetaAngle < (3 * pi) / 4:
                    # Correct 2d point will be one with maximum Y-Value
                    for points2d in allIntersectionPoints2d:
                        xOrYList.append(points2d[1, 0])
                    correct2dPointArray = allIntersectionPoints2d[xOrYList.index(max(xOrYList))]
                elif thetaAngle <= (-(3 * pi)) / 4 or thetaAngle >= (3 * pi) / 4:
                    # Correct 2d point will be one with minimum X-Value
                    for points2d in allIntersectionPoints2d:
                        xOrYList.append(points2d[0, 0])
                    correct2dPointArray = allIntersectionPoints2d[xOrYList.index(min(xOrYList))]
                elif (-(3 * pi)) / 4 <= thetaAngle and thetaAngle <= (-pi) / 4:
                    # Correct 2d point will be one with minimum Y-Value
                    for points2d in allIntersectionPoints2d:
                        xOrYList.append(points2d[1, 0])
                    correct2dPointArray = allIntersectionPoints2d[xOrYList.index(min(xOrYList))]
                elif thetaAngle == 9999:
                    # Special case for Original plane being parallel to X-Y Plane.
                    # We have decided correct 2d point will be one with minimum X-Value
                    for points2d in allIntersectionPoints2d:
                        xOrYList.append(points2d[0, 0])
                    correct2dPointArray = allIntersectionPoints2d[xOrYList.index(min(xOrYList))]

                # Find correct starting point in actual plane and also find index of edge containing it
                for key in list(intersection2dPoints.keys()):
                    pointList2d = intersection2dPoints[key]
                    for i, point2dArray in enumerate(pointList2d):
                        # print 'Th: ', thetaAngle
                        if (point2dArray == correct2dPointArray).all():
                            indexOfEdgeContainingFinal3dStartingPoint = key
                            final3dStartingPoint = array(
                                [[intersectPointsValues[key][i].X()], [intersectPointsValues[key][i].Y()], [intersectPointsValues[key][i].Z()]])
                            # Break Inner Loop
                            break
                    else:
                        # Continue if inner loop not broken
                        continue
                    # Break outer loop as inner loop was broken
                    break

                # Check whether correct starting point is already in the list
                points3d = deepcopy(edgePointDict[indexOfEdgeContainingFinal3dStartingPoint])
                nPoints = points3d.shape[1]
                IntersectionAlreadyInPointList = False
                needToSplitEdge = True
                for j in range(nPoints):
                    if (abs(points3d[0, j] - final3dStartingPoint[0, 0]) < 0.0001) and (
                            abs(points3d[1, j] - final3dStartingPoint[1, 0]) < 0.0001) and (
                            abs(points3d[2, j] - final3dStartingPoint[2, 0]) < 0.0001):
                        IntersectionAlreadyInPointList = True
                        # Starting point is already in the point list. Check if it`s luckily at the start or end of edge
                        if j == 0:
                            # No need to spit edge
                            needToSplitEdge = False
                            # Reorder crvOrder accordingly
                            crvOrder = ReorderList(crvOrder,
                                                   crvOrder[
                                                       crvOrder.index(indexOfEdgeContainingFinal3dStartingPoint)])

                        elif j == nPoints - 1 and crvOrder[
                            len(crvOrder) - 1] != indexOfEdgeContainingFinal3dStartingPoint:
                            # Next edge will be the starting edge
                            needToSplitEdge = False
                            # Reorder crvOrder accordingly
                            crvOrder = ReorderList(crvOrder,
                                                   crvOrder[crvOrder.index(
                                                       indexOfEdgeContainingFinal3dStartingPoint) + 1])

                        # Starting point is already in the point list with index j which is not start or finish
                        else:
                            # Split the point, normal and tangent arrays into two parts at intersection point
                            splittedPointsArrays = hsplit(edgePointDict[indexOfEdgeContainingFinal3dStartingPoint],
                                                          [j + 1, nPoints])
                            splittedNormalsArrays = hsplit(
                                edgeNormalDict[indexOfEdgeContainingFinal3dStartingPoint],
                                [j + 1, nPoints])
                            splittedTangentsArrays = hsplit(
                                edgeTangentDict[indexOfEdgeContainingFinal3dStartingPoint],
                                [j + 1, nPoints])
                            splittedFuvsArrays = hsplit(edgeFuvDict[indexOfEdgeContainingFinal3dStartingPoint], [j + 1, nPoints])
                            splittedCurvatureArrays = hsplit(edgeCurvatureDict[indexOfEdgeContainingFinal3dStartingPoint], [j + 1, nPoints])
                            if splittedPointsArrays[0].size != 0:
                                edgePointDict[indexOfEdgeContainingFinal3dStartingPoint] = splittedPointsArrays[0]
                                edgeNormalDict[indexOfEdgeContainingFinal3dStartingPoint] = splittedNormalsArrays[0]
                                edgeTangentDict[indexOfEdgeContainingFinal3dStartingPoint] = splittedTangentsArrays[0]
                                edgeFuvDict[indexOfEdgeContainingFinal3dStartingPoint] = splittedFuvsArrays[0]
                                edgeCurvatureDict[indexOfEdgeContainingFinal3dStartingPoint] = splittedCurvatureArrays[0]

                            # Create a new edge and store points normals and tangents for it along with intersection point,normal and tangent added at beginning
                            if splittedPointsArrays[1].size != 0:
                                edgePointDict[edgeIndex] = append(edgePointDict[indexOfEdgeContainingFinal3dStartingPoint][:, j:j + 1],
                                                                  splittedPointsArrays[1], axis=1)
                                edgeNormalDict[edgeIndex] = append(edgeNormalDict[indexOfEdgeContainingFinal3dStartingPoint][:, j:j + 1],
                                                                   splittedNormalsArrays[1], axis=1)
                                edgeTangentDict[edgeIndex] = append(edgeTangentDict[indexOfEdgeContainingFinal3dStartingPoint][:, j:j + 1],
                                                                    splittedTangentsArrays[1], axis=1)
                                edgeFuvDict[edgeIndex] = append(edgeFuvDict[indexOfEdgeContainingFinal3dStartingPoint][:, j:j + 1],
                                                                splittedFuvsArrays[1], axis=1)
                                edgeCurvatureDict[edgeIndex] = append(edgeCurvatureDict[indexOfEdgeContainingFinal3dStartingPoint][:, j:j + 1],
                                                                      splittedCurvatureArrays[1], axis=1)

                            # Add the newly created edge to the crvOrder
                            crvOrder.insert((crvOrder.index(indexOfEdgeContainingFinal3dStartingPoint) + 1), edgeIndex)
                            # Reorder the crvOrder so our new edge is at the beginning
                            crvOrder = ReorderList(crvOrder, edgeIndex)
                # Find the neighboring points to the correct starting point in the edge if correct starting point is not in the list
                if not IntersectionAlreadyInPointList:
                    v1 = array([points3d[0, 0:(nPoints - 1)] - final3dStartingPoint[0, 0],
                                points3d[1, 0:(nPoints - 1)] - final3dStartingPoint[1, 0],
                                points3d[2, 0:(nPoints - 1)] - final3dStartingPoint[2, 0]])
                    v2 = array([points3d[0, 1:nPoints] - final3dStartingPoint[0, 0],
                                points3d[1, 1:nPoints] - final3dStartingPoint[1, 0],
                                points3d[2, 1:nPoints] - final3dStartingPoint[2, 0]])
                    dotArray = array([v1.transpose()[:, 0] * v2.transpose()[:, 0] + v1.transpose()[:,
                                                                                    1] * v2.transpose()[:,
                                                                                         1] + v1.transpose()[:,
                                                                                              2] * v2.transpose()[:,
                                                                                                   2]])
                    indexOfPrevPointToStartingPoint = dotArray.argmin(axis=1)

                    # Find the face that this edge belonged to
                    face = TopoDS.topods_Face(TopoDS.TopoDS_Shape())
                    if section.HasAncestorFaceOn1(edgeMarkers[indexOfEdgeContainingFinal3dStartingPoint], face):
                        aNewFaceIndex = -1
                        # Get face index in the workpiece faces
                        try:
                            for anObject in session.objects:
                                for ijk, aFace in enumerate(anObject.faces):
                                    if aFace == face:
                                        aNewFaceIndex = ijk
                                        break
                                if aNewFaceIndex != -1:
                                    break
                        except:
                            for aShape in extraShapes:
                                ijk = 0
                                for aExtraFace in Topo(aShape).faces():
                                    if aExtraFace == face:
                                        aNewFaceIndex = ijk
                                        break
                                    ijk += 1
                                if aNewFaceIndex != -1:
                                    break

                        surf = BRepLProp_SLProps(BRepAdaptor_Surface(face), 1, 0.000001)
                        # Extract UV Curve
                        curve2D, uMin2D, uMax2D = BRep_Tool_CurveOnSurface(edgeMarkers[indexOfEdgeContainingFinal3dStartingPoint], face)
                        curve2D = curve2D
                        # Extract 3D Curve
                        curve3DHandle, uMin, uMax = BRep_Tool_Curve(edgeMarkers[indexOfEdgeContainingFinal3dStartingPoint])
                        curve3D = curve3DHandle
                        cLen = edgePointDict[indexOfEdgeContainingFinal3dStartingPoint].shape[1]
                        # Compute Curve Points
                        uRef = linspace(uMin, uMax, cLen)
                        uRef2D = linspace(uMin2D, uMax2D, cLen)
                        k = indexOfPrevPointToStartingPoint
                        u1 = uRef[k]
                        u2 = uRef[k + 1]
                        u1u2XYZDistance = sqrt((points3d[0, k + 1] - points3d[0, k]) ** 2 + (
                                points3d[1, k + 1] - points3d[1, k]) ** 2 + (
                                                       points3d[2, k + 1] - points3d[2, k]) ** 2)
                        u1ToStartPointDistance = sqrt((final3dStartingPoint[0, 0] - points3d[0, k]) ** 2 + (
                                final3dStartingPoint[1, 0] - points3d[1, k]) ** 2 + (
                                                              final3dStartingPoint[2, 0] - points3d[2, k]) ** 2)
                        uStartPoint = u1 + (u2 - u1) * (u1ToStartPointDistance / u1u2XYZDistance)
                        uStartPoint2d = uRef2D[k] + (uRef2D[k + 1] - uRef2D[k]) * (
                                u1ToStartPointDistance / u1u2XYZDistance)
                        # Eval Pnt and Tangent
                        curve3D.D1(uStartPoint[0], evalPnt, evalTan)
                        # Normalize the tangent
                        evalTan = evalTan.Normalized()
                        # Eval Normal
                        curve2D.D0(uStartPoint2d[0], evalPnt2d)
                        # Eval Curvature
                        glpClp = GeomLProp_CLProps(curve3DHandle, 2, 0.0001)
                        glpClp.SetParameter(uStartPoint[0])
                        try:
                            aCurvature = glpClp.Curvature()
                        except:
                            aCurvature = 0.0

                        try:
                            surf.SetParameters(evalPnt2d.X(), evalPnt2d.Y())
                            evalNor = surf.Normal()
                        except:
                            try:
                                point3d = gp_Pnt()
                                uTan = gp_Vec()
                                vTan = gp_Vec()
                                surfHandle = BRep_Tool_Surface(face)
                                surfHandle.D1(evalPnt2d.X(), evalPnt2d.Y(), point3d, uTan, vTan)
                                uTan.Normalize()
                                vTan.Normalize()
                                evalNor = gp_Dir(uTan.Crossed(vTan))
                            except:
                                evalNor = gp_Dir(0, 0, 1)
                        if face.Orientation() == TopAbs_REVERSED or evalNor.Z() < -1:
                            evalNor.Reverse()
                        loopStartPoint3d = array([[evalPnt.X()], [evalPnt.Y()], [evalPnt.Z()]])
                        loopStartPoint3dTan = array([[evalTan.X()], [evalTan.Y()], [evalTan.Z()]])
                        loopStartPoint3dNor = array([[evalNor.X()], [evalNor.Y()], [evalNor.Z()]])
                        loopStartPoint3dFuv = array([[aNewFaceIndex], [evalPnt2d.X()], [evalPnt2d.Y()]])
                        loopStartPoint3dCurvature = array([[aCurvature], [aCurvature]])

                    # Split the point, normal and tangent arrays into two parts at intersection point
                    splittedPointsArrays = hsplit(edgePointDict[indexOfEdgeContainingFinal3dStartingPoint],
                                                  [int(indexOfPrevPointToStartingPoint) + 1, nPoints])
                    splittedNormalsArrays = hsplit(edgeNormalDict[indexOfEdgeContainingFinal3dStartingPoint],
                                                   [int(indexOfPrevPointToStartingPoint) + 1, nPoints])
                    splittedTangentsArrays = hsplit(edgeTangentDict[indexOfEdgeContainingFinal3dStartingPoint],
                                                    [int(indexOfPrevPointToStartingPoint) + 1, nPoints])
                    splittedFuvsArrays = hsplit(edgeFuvDict[indexOfEdgeContainingFinal3dStartingPoint],
                                                [int(indexOfPrevPointToStartingPoint) + 1, nPoints])
                    splittedCurvatureArrays = hsplit(edgeCurvatureDict[indexOfEdgeContainingFinal3dStartingPoint],
                                                     [int(indexOfPrevPointToStartingPoint) + 1, nPoints])
                    edgePointDict[indexOfEdgeContainingFinal3dStartingPoint] = append(splittedPointsArrays[0], final3dStartingPoint, axis=1)
                    edgeNormalDict[indexOfEdgeContainingFinal3dStartingPoint] = append(splittedNormalsArrays[0], loopStartPoint3dNor, axis=1)
                    edgeTangentDict[indexOfEdgeContainingFinal3dStartingPoint] = append(splittedTangentsArrays[0], loopStartPoint3dTan, axis=1)
                    edgeFuvDict[indexOfEdgeContainingFinal3dStartingPoint] = append(splittedFuvsArrays[0], loopStartPoint3dFuv, axis=1)
                    edgeCurvatureDict[indexOfEdgeContainingFinal3dStartingPoint] = append(splittedCurvatureArrays[0], loopStartPoint3dCurvature,
                                                                                          axis=1)

                    # Create a new edge and store points normals and tangents for it along with intersection point,normal and tangent added at beginning
                    edgePointDict[edgeIndex] = append(final3dStartingPoint, splittedPointsArrays[1], axis=1)
                    edgeNormalDict[edgeIndex] = append(loopStartPoint3dNor, splittedNormalsArrays[1], axis=1)
                    edgeTangentDict[edgeIndex] = append(loopStartPoint3dTan, splittedTangentsArrays[1], axis=1)
                    edgeFuvDict[edgeIndex] = append(loopStartPoint3dFuv, splittedFuvsArrays[1], axis=1)
                    edgeCurvatureDict[edgeIndex] = append(loopStartPoint3dCurvature, splittedCurvatureArrays[1], axis=1)

                    # Add the newly created edge to the crvOrder
                    crvOrder.insert((crvOrder.index(indexOfEdgeContainingFinal3dStartingPoint) + 1), edgeIndex)

                    # Reorder the crvOrder so our new edge is at the beginning
                    crvOrder = ReorderList(crvOrder, edgeIndex)
        # Finding correct starting point and splitting the edge containing it and reordering crvOrder ends

        # # Check orientation for open loop:
        # if not wire.Closed() and crvOrder:
        #     normalVec = array([direction.X(), direction.Y(), direction.Z()])
        #
        #     # If normal of original plane already in Z-direction, set a value for thetaAngle to be identifiable
        #     if normalVec[0] == 0.00 and normalVec[1] == 0.00 and normalVec[2] != 0.00:
        #         # Given a separate value
        #         thetaAngle = 9999
        #
        #     else:
        #         # Project normal vector on X-Y plane (For else case) and find it's angle with positive X-Axis
        #         nVec = array([0, 0, 1])
        #         dotProd = dot(normalVec, nVec)
        #         proj2dVec = array(
        #             [(direction.X() - (dotProd * nVec[0])),
        #              (direction.Y() - (dotProd * nVec[1]))])  # ignore z value
        #         xVec = array([1, 0])  # Ignore z, as we take 2d plane
        #         dotProduct = xVec[0] * proj2dVec[0] + xVec[1] * proj2dVec[1]  # dot product
        #         det = xVec[0] * proj2dVec[1] - xVec[1] * proj2dVec[0]  # determinant
        #         thetaAngle = atan2(det, dotProduct)
        #
        #     openLoopFlip = False
        #     openLoopStartPoint = edgePointDict[crvOrder[0]][:, 0]
        #     openLoopEndPoint = edgePointDict[crvOrder[len(crvOrder) - 1]][:,
        #                        edgePointDict[crvOrder[len(crvOrder) - 1]].shape[1] - 1]
        #     if (-pi) / 4 <= thetaAngle <= pi / 4:
        #         # Correct 2d point will be one with maximum X-Value
        #         if openLoopStartPoint[1] < openLoopEndPoint[1]:
        #             openLoopFlip = True
        #     elif pi / 4 < thetaAngle < (3 * pi) / 4:
        #         # Correct 2d point will be one with maximum Y-Value
        #         if openLoopEndPoint[0] < openLoopStartPoint[0]:
        #             openLoopFlip = True
        #     elif thetaAngle <= (-(3 * pi)) / 4 and thetaAngle >= (3 * pi) / 4:
        #         # Correct 2d point will be one with minimum X-Value
        #         if openLoopEndPoint[1] < openLoopStartPoint[1]:
        #             openLoopFlip = True
        #     elif (-(3 * pi)) / 4 <= thetaAngle <= (-pi) / 4:
        #         # Correct 2d point will be one with minimum Y-Value
        #         if openLoopStartPoint[0] < openLoopEndPoint[0]:
        #             openLoopFlip = True
        #     elif thetaAngle == 9999:
        #         # Correct 2d point will be one with minimum X-Value as decided if original plane is parallel to X-Y plane
        #         if openLoopEndPoint[1] < openLoopStartPoint[1]:
        #             openLoopFlip = True
        #
        #     # Flip the points, normals and tangents on the loop
        #     if openLoopFlip:
        #         crvOrder = crvOrder[::-1]
        #         for correctIndex in crvOrder:
        #             edgePointDict[correctIndex] = fliplr(edgePointDict[correctIndex])
        #             edgeNormalDict[correctIndex] = fliplr(edgeNormalDict[correctIndex])
        #             edgeTangentDict[correctIndex] = -fliplr(edgeTangentDict[correctIndex])
        #             edgeFuvDict[correctIndex] = fliplr(edgeFuvDict[correctIndex])
        #             edgeCurvatureDict[correctIndex] = fliplr(edgeCurvatureDict[correctIndex])

        # Add the points to the final points array in correct order
        # for correctIndex in origCrvOrder:
        for correctIndex in crvOrder:
            try:
                points = append(points, edgePointDict[correctIndex], axis=1)
                normals = append(normals, edgeNormalDict[correctIndex], axis=1)
                tangents = append(tangents, edgeTangentDict[correctIndex], axis=1)
                fuv = append(fuv, edgeFuvDict[correctIndex], axis=1)
                curvature = append(curvature, edgeCurvatureDict[correctIndex], axis=1)
            except:
                pass

    # Find nans (Samyak tried to kill me!!!!!!)
    notNans = sum(isnan(points) | isnan(normals) | isnan(tangents), 0) == 0
    points = points[:, notNans]
    normals = normals[:, notNans]
    tangents = tangents[:, notNans]
    fuv = fuv[:, notNans]
    curvature = curvature[:, notNans]

    # NEW FEATURE ADDED - curvature based point spacing
    if pointMode == 1 and trackSpacing and len(curvature[0]) > 0 and sum(curvature[0]) != 0.:
        # Get the number points for this track
        pLen = len(points[0])
        if selectedArea > 0.0:
            totalPnts = ceil(selectedArea / (trackSpacing * sampling))
            pDiff = insert((points[:, 0:pLen - 1] - points[:, 1:pLen]), 0, 0, axis=1)
            deltas = sqrt(pDiff[0, :] ** 2 + pDiff[1, :] ** 2 + pDiff[2, :] ** 2)
            pLen = ceil(((sum(deltas) * trackSpacing) / selectedArea) * totalPnts)

        # Compute cumulative sum for sqrt of curvature
        curvSum = cumsum(sqrt(curvature[0]))

        # Smooth cumulative curve
        b = gaussian(6, 2)
        curvSum = filters.convolve1d(curvSum, b / b.sum())

        # Redistribute points according to prominence of curvature
        newDistribution = linspace(0, curvSum[-1], pLen)
        x = interp(newDistribution, curvSum, points[0])
        y = interp(newDistribution, curvSum, points[1])
        z = interp(newDistribution, curvSum, points[2])
        points = array([x, y, z])

        nx = interp(newDistribution, curvSum, normals[0])
        ny = interp(newDistribution, curvSum, normals[1])
        nz = interp(newDistribution, curvSum, normals[2])
        normals = array([nx, ny, nz])

        tx = interp(newDistribution, curvSum, tangents[0])
        ty = interp(newDistribution, curvSum, tangents[1])
        tz = interp(newDistribution, curvSum, tangents[2])

        # Filter for removing spurious tangents generated at the edge of sections (not fully tested).
        mt = mean([tx, ty, tz], 1)
        for ind in range(len(tx)):
            if dot(mt, array([tx[ind], ty[ind], tz[ind]])) < 0:
                tx[ind] = -tx[ind]
                ty[ind] = -ty[ind]
                tz[ind] = -tz[ind]
        tangents = array([tx, ty, tz])

        xfuv = interp(newDistribution, curvSum, fuv[0])
        yfuv = interp(newDistribution, curvSum, fuv[1])
        zfuv = interp(newDistribution, curvSum, fuv[2])
        fuv = array([xfuv, yfuv, zfuv])

    # TODO - Check and try to automatically fix normals
    pLen = len(normals[0])
    if pLen > 0 and not silent:
        # Flip 1st normal vector if it is in -ve z-direction
        if normals[2][0] < 0.0 < max(normals[2][1:5]):
            # Let user know about it and ask if the user wants to automatically fix it
            if autoFixNormals is False:
                argData = [None,
                           "Issues were detected where surface normal seems to be changing drastically. Do you want it to be fixed automatically ? " +
                           "\nEven if you click Yes, WE STRONGLY ADVISE YOU TO DOUBLE CHECK THE NORMALS AND DO COLLISION CHECK IN THE SIMULATOR.",
                           'Please Select']
                mData = [argData, {}]
                result = YesNoDlg(title='', data=mData)
                if result is False:
                    autoFixNormals = True
                    normals[:, 0] *= -1.0
        normalError = 0
        for i in range(normals.shape[1] - 1):
            if dot(normals[:, i], normals[:, i + 1]) <= -0.3:
                if i == 0:  # Special case first point
                    normals[:, 0] = -normals[:, 0]
                    continue
                if i + 1 == normals.shape[1] - 1 and normalError == 0:  # Special case last point
                    normals[:, i + 1] = -normals[:, i + 1]
                    continue
                normalError += 1
        if normalError > 0:
            newNormals = zeros((3, pLen))
            newNormals[:, 0] = normals[:, 0]
            if autoFixNormals == False:
                argData = [None,
                           "Issues were detected where surface normal seems to be changing drastically. Do you want it to be fixed automatically ? " +
                           "\nEven if you click Yes, WE STRONGLY ADVISE YOU TO DOUBLE CHECK THE NORMALS AND DO COLLISION CHECK IN THE SIMULATOR.",
                           'Please Select']
                mData = [argData, {}]
                result = YesNoDlg(title='', data=mData)
                if result is False:
                    autoFixNormals = True
            if autoFixNormals:
                for i in range(normals.shape[1] - 1):
                    newNormals[:, i + 1] = normals[:, i + 1]
                    if dot(newNormals[:, i], normals[:, i + 1]) < 0.0:
                        # If normal changes more than 90 degree, flip them automatically to fix the potential issue
                        newNormals[:, i + 1] = -1 * normals[:, i + 1]
                normals = newNormals

    # Mask to remove points that are very close to each other / repeated points
    pLen = len(points[0])
    if pLen > 0:
        pDiff = insert((points[:, 0:pLen - 1] - points[:, 1:pLen]), 0, 0, axis=1)
        deltas = sqrt(pDiff[0, :] ** 2 + pDiff[1, :] ** 2 + pDiff[2, :] ** 2)
        deltas[0] = deltas[1]
        mask = deltas >= 0.0001
        points = points[:, mask]
        normals = normals[:, mask]
        tangents = tangents[:, mask]
        fuv = fuv[:, mask]
    # pr.disable()
    # s = StringIO.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print s.getvalue()

    return points, normals, tangents, fuv
    
def section2pointsOLD(section, direction, sampling):

    """!
    ?
    @param section:
    @param direction:
    @param sampling:
    @return: tuple of points, normals, tangents
    """

    # Build section
    section.Approximation(True)
    section.ComputePCurveOn1(True)
    #section.ComputePCurveOn2(True)
    section.Build()

    # List for collecting all edges from section result
    edgeList = TopTools.TopTools_HSequenceOfShape()
    for edge in Topo(section.Shape()).edges():
        edgeList.Append(edge)

    # List of multiple wires generated by ConnectEdgesToWires algorithm
    wireList = TopTools.TopTools_HSequenceOfShape()
    wireListHandle = wireList
    ShapeAnalysis_FreeBounds_ConnectEdgesToWires(edgeList, 1e-2, False, wireListHandle)

    #Iterating through each wire and then through the edges in each wire
    wires = []
    for wireIndex in range(1,wireListHandle.Length()+1):
        wire = wireListHandle.Value(wireIndex)

        # Wire Healing Tool
        sewd = ShapeExtend_WireData()
        sfw = ShapeFix_Wire()
        exp = BRepTools_WireExplorer(TopoDS.topods_Wire(wire))
        while exp.More():
            edge = exp.Current()
            sewd.Add(edge)
            sfw.Load(sewd)
            sfw.Perform()
            sfw.FixReorder()
            sfw.SetMaxTolerance(0.1)
            sfw.SetClosedWireMode(True)
            sfw.FixConnected(0.1)
            sfw.FixClosed(0.1)
            exp.Next()
        wires.append(sfw.Wire())

    # Connect Wires to Wire (Test)
    openWireList = TopTools.TopTools_HSequenceOfShape()
    closedWireList = TopTools.TopTools_HSequenceOfShape()
    newWires = []
    if wireListHandle.Length() > 1:
        for wire in wires:
            if not wire.Closed():
                openWireList.Append(wire)
            else:
                newWires.append(wire)
    else:
        for wire in wires:
            newWires.append(wire)
    tol = 0.1
    wireWire = False
    while wireListHandle.Length() > 1:
        wireList = TopTools.TopTools_HSequenceOfShape()
        wireListHandle = wireList
        ShapeAnalysis_FreeBounds_ConnectWiresToWires(openWireList, tol, False, wireListHandle)
        tol = tol + 0.1
    if openWireList.Length() > 1 and wireListHandle.Length() == 1:
        newWires.append(wireListHandle.Value(1))
    # Test Work Done

    ## HACK - Check newWires is empty or not
    if len(newWires) == 0 and edgeList.Length != 0:
        for edge in Topo(section.Shape()).edges():
            aWire = BRepBuilderAPI_MakeWire()
            aWire.Add(edge)
            newWires.append(aWire.Wire())


    ############################
    # Perform Sampling of Wires
    points = array([[],[],[]]);
    normals = array([[],[],[]]);
    tangents = array([[],[],[]]);
    markers = []
    
    # Initialize OCC vars
    evalPnt = gp_Pnt()
    evalPnt2d = gp_Pnt2d() 
    evalTan = gp_Vec()

    # Loop through wires
    for wire in newWires:
        # Dictionary for storing crvPoints, crvNormals, crvTangents for correcting their order later
        edgeIndex = 0
        edgePointDict = {}
        edgeNormalDict = {}
        edgeTangentDict = {}

        # Explore wire (to get edges)
        exp = BRepTools_WireExplorer(TopoDS.topods_Wire(wire))
        while exp.More():
            # Init edge and face
            edge = exp.Current()
            face = TopoDS.topods_Face(TopoDS.TopoDS_Shape())
            # Hack for HasAncestorFaceOn1 error (look-up original edge handle)
            if section.HasAncestorFaceOn1(edge, face) == 0:
                curve = BRepAdaptor_Curve(edge)
                startPoint = curve.Value(curve.FirstParameter())
                endPoint = curve.Value(curve.LastParameter())
                for anedge in Topo(section.Shape()).edges():
                    acurve = BRepAdaptor_Curve(anedge)
                    aStartPoint = acurve.Value(acurve.FirstParameter())
                    anEndPoint = acurve.Value(acurve.LastParameter())
                    if (startPoint == aStartPoint and endPoint == anEndPoint):
                        edge = anedge
                        break

            # Look-up parent Face
            if section.HasAncestorFaceOn1(edge, face):
                # Get Geometric Surface
                surfHandle = BRep_Tool_Surface(face)
                surf = BRepLProp_SLProps(BRepAdaptor_Surface(face), 1, 0.000001)

                # Extract UV Curve
                curve2D, uMin2D, uMax2D = BRep_Tool_CurveOnSurface(edge, face)

                # Extract 3D Curve
                curve3D, uMin, uMax = BRep_Tool_Curve(edge)

                # Get Linear Properties
                props = GProp_GProps()
                brepgprop_LinearProperties(edge, props)
                length = props.Mass()

                # Initialize Numpy Curves
                cLen = max(int(ceil(length/sampling)), 3)
                crvPoints = zeros((3,cLen))
                crvNormals = zeros((3,cLen))
                crvTangents = zeros((3,cLen))

                # Compute Curve Points
                uRef = linspace(uMin, uMax, cLen)
                uRef2D = linspace(uMin2D, uMax2D, cLen)
                for u in range(cLen):
                    # Eval Point and Tangent
                    curve3D.D1(uRef[u], evalPnt, evalTan)

                    # Old algorithm (slow)
                    projection = GeomAPI_ProjectPointOnSurf(evalPnt, surfHandle)
                    evalU, evalV = projection.LowerDistanceParameters()                    
                    surf.SetParameters(evalU, evalV)

                    # Eval Normal
                    #curve2D.D0(uRef2D[u], evalPnt2d)
                    #surf.SetParameters(evalPnt2d.X(), evalPnt2d.Y())
                    evalNor = surf.Normal()
                    if face.Orientation() == TopAbs_REVERSED:
                        evalNor.Reverse()

                    # Save in NumPy format
                    crvPoints[0,u] = evalPnt.X()
                    crvPoints[1,u] = evalPnt.Y()
                    crvPoints[2,u] = evalPnt.Z()
                    magnitude = sqrt(evalNor.X()**2 + evalNor.Y()**2 +  evalNor.Z()**2)
                    crvNormals[0,u] = evalNor.X() / magnitude
                    crvNormals[1,u] = evalNor.Y() / magnitude
                    crvNormals[2,u] = evalNor.Z() / magnitude
                    magnitude = sqrt(evalTan.X()**2 + evalTan.Y()**2 +  evalTan.Z()**2)
                    crvTangents[0,u] = evalTan.X() / magnitude
                    crvTangents[1,u] = evalTan.Y() / magnitude
                    crvTangents[2,u] = evalTan.Z() / magnitude

                # Check Tangent Direction
                crvDir = crvPoints[:,1:cLen] - crvPoints[:,0:cLen-1]
                crvDir = append(crvDir, crvDir[:,cLen-2:cLen-1], axis=1)
                for n in range(cLen):
                    if dot(crvDir[:,n], crvTangents[:,n]) < 0:
                        crvTangents[:,n] = -crvTangents[:,n]

                # Create dictionary for points, normals, tangents and edges
                edgePointDict[edgeIndex] = crvPoints
                edgeNormalDict[edgeIndex] = crvNormals
                edgeTangentDict[edgeIndex] = crvTangents
                edgeIndex = edgeIndex + 1

                # Save edge in markers (for display)
                markers.append(edge)

            exp.Next()

        # The original edge/curve order given by BRep_WireExplorer
        origCrvOrder = []
        for i in range(edgeIndex):
            origCrvOrder.append(i)

        # Check for duplicate edges and remove them if found
        copyEdgePointDict = {}
        duplicateList = []
        for i in range(edgeIndex):
            found = False
            if len(copyEdgePointDict) == 0:
                copyEdgePointDict[i] = edgePointDict[i]
            else:
                for j in range(len(copyEdgePointDict)):
                    if j in duplicateList:
                        continue
                    if edgePointDict[i].shape[1] == copyEdgePointDict[j].shape[1]:
                        if array_equal(edgePointDict[i], copyEdgePointDict[j]):
                            found = True
                            duplicateList.append(i)
                            break
                        elif array_equal(edgePointDict[i], fliplr(copyEdgePointDict[j])):
                            found = True
                            duplicateList.append(i)
                            break
                if found == False:
                    copyEdgePointDict[i] = edgePointDict[i]
        # End of Duplicate edges removal

        # Check edges for: order, flip them if required
        crvOrder = []
        tol = 0.00001
        while len(crvOrder) < (edgeIndex - len(duplicateList)):
            if len(crvOrder) == 0:
                crvOrder.append(0)
                if edgeIndex > 1:
                    if (dot(edgePointDict[0][:,0]-edgePointDict[1][:,0],edgePointDict[0][:,0]-edgePointDict[1][:,0]) < 0.001) or \
                            (dot(edgePointDict[0][:,0]-edgePointDict[1][:,edgePointDict[1].shape[1]-1],edgePointDict[0][:,0]-edgePointDict[1][:,edgePointDict[1].shape[1]-1]) < 0.001):
                        edgePointDict[0] = fliplr(edgePointDict[0])
                        edgeNormalDict[0] = fliplr(edgeNormalDict[0])
                        edgeTangentDict[0] = -fliplr(edgeTangentDict[0])
                currentCurve = edgePointDict[0]
                currCurveLen = edgePointDict[0].shape[1]
            else:
                checkLength = len(crvOrder)
                for i in range(edgeIndex):
                    if i in crvOrder or i in duplicateList:
                        continue
                    cLen = edgePointDict[i].shape[1]
                    if (abs(edgePointDict[i][0,0] - currentCurve[0,currCurveLen-1]) < tol) and \
                            (abs(edgePointDict[i][1,0] - currentCurve[1,currCurveLen-1]) < tol) and \
                            (abs(edgePointDict[i][2,0] - currentCurve[2,currCurveLen-1]) < tol):
                        crvOrder.append(i)
                        currentCurve = edgePointDict[i]
                        currCurveLen = edgePointDict[i].shape[1]
                        break
                    elif (abs(edgePointDict[i][0,cLen-1] - currentCurve[0,currCurveLen-1]) < tol) and \
                            (abs(edgePointDict[i][1,cLen-1] - currentCurve[1,currCurveLen-1]) < tol) and \
                             (abs(edgePointDict[i][2,cLen-1] - currentCurve[2,currCurveLen-1]) < tol):
                        crvOrder.append(i)
                        edgePointDict[i] = fliplr(edgePointDict[i])
                        edgeNormalDict[i] = fliplr(edgeNormalDict[i])
                        edgeTangentDict[i] = -fliplr(edgeTangentDict[i])
                        currentCurve = edgePointDict[i]
                        currCurveLen = edgePointDict[i].shape[1]
                        break
                if checkLength == len(crvOrder):
                    tol = tol + 0.001
                else:
                    tol = 0.0001

        # Check CCW orientation (for closed loops)
        if wire.Closed():
            # Find the longest edge (edge with maximum points)
            edgeLengthList = []
            for m in range(edgeIndex):
                edgeLengthList.append(edgePointDict[m].shape[1])
            if not edgeLengthList:
                continue
            maxValue = max(edgeLengthList)
            maxIndex = edgeLengthList.index(maxValue)

            # Find mid-point of longest edge
            midIndexInCrvPoints = maxValue//2
            crvPointsLongestEdge = edgePointDict[maxIndex]
            midPointOnCrvPoints = crvPointsLongestEdge[:,midIndexInCrvPoints]

            # Compute cross product of direction and tangent, then project from mid-point
            crvTangentLongestEdge = edgeTangentDict[maxIndex]
            a = array([direction.X(), direction.Y(), direction.Z()])
            b = crvTangentLongestEdge[:,midIndexInCrvPoints]
            crossProduct = cross(a,b)
            InPointToTest = midPointOnCrvPoints + 0.01*crossProduct

            # Find Rotation Matrix for rotating plane to align it with xy Plane
            tempPoints = array([[],[],[]]);
            if a[0] == 0.00 and a[1] == 0.00 and a[2] != 0:
                # Store all the points in tempPoints
                for correctIndex in crvOrder:
                    tempPoints = append(tempPoints, edgePointDict[correctIndex], axis=1)
                # Delete the Z-coordinate to get 2-d planar points
                temp2dPoints = delete(tempPoints, (2), axis=0)

                # Set RotMatrix as Identity
                RotMatrix = array([[ 1,  0,  0 ],
                                   [ 0,  1,  0 ],
                                   [ 0,  0,  1 ]])
            else:
                # Unit Vector in Z-dir
                b = array([0,0,1])
                # Get rotation axis
                axis = cross(b,Normalize(a))
                # Get rotation angle
                theta = acos((dot(a,b))/(linalg.norm(a)*linalg.norm(b)))
                # Get rotation matrix
                RotMatrix = RotationAxis(axis, theta)

                # Get rotated points on the new plane parallel to x-y plane
                for i, correctIndex in enumerate(crvOrder):
                    correctIndex = abs(correctIndex)
                    anArray = array([[],[],[]]);
                    for j in range(edgePointDict[correctIndex].shape[1]):
                        anArray = append(anArray, array([[RotMatrix.dot(edgePointDict[correctIndex][:,j])[0]],[RotMatrix.dot(edgePointDict[correctIndex][:,j])[1]],[RotMatrix.dot(edgePointDict[correctIndex][:,j])[2]]]), axis=1)
                    # Store points in tempPoints for Path method of matplotlib to check whether the InPointToTest lies inside or outside the loop
                    tempPoints = append(tempPoints, anArray, axis=1)

                # Remove the Z-coordinate from them as they are planar points
                temp2dPoints = delete(tempPoints, (2), axis=0)

                # Rotate the InPointToTest on the new plane
                InPointToTest = RotMatrix.dot(InPointToTest)

            # Check if point lies inside or outside the path
            InPointToTest = array([[InPointToTest[0]],[InPointToTest[1]]])
            p = Path(temp2dPoints.transpose())
            contain = False
            if p.contains_point(InPointToTest) == 1:
                contain = True
            if not contain:
                # Change orientation (Flip points, normals and tangents)
                crvOrder = crvOrder[::-1]
                for index in crvOrder:
                    edgePointDict[index] = fliplr(edgePointDict[index])
                    edgeNormalDict[i] = fliplr(edgeNormalDict[i])
                    edgeTangentDict[i] = -fliplr(edgeTangentDict[i])
                # Flip the tempPoints and 2d points
                tempPoints = fliplr(tempPoints)
                temp2dPoints = fliplr(temp2dPoints)

        # Find the correct starting point and the edge containing it if wire is closed
        if wire.Closed():
            intersectPointsValues = {}
            normalVec = array([direction.X(), direction.Y(), direction.Z()])

            # Find centre point of bounding box of curve
            xMin = temp2dPoints.transpose()[argmin(temp2dPoints.transpose()[:, 0])][0]
            xMax = temp2dPoints.transpose()[argmax(temp2dPoints.transpose()[:, 0])][0]
            yMin = temp2dPoints.transpose()[argmin(temp2dPoints.transpose()[:, 1])][1]
            yMax = temp2dPoints.transpose()[argmax(temp2dPoints.transpose()[:, 1])][1]
            xCentre = (xMin + xMax)*0.5
            yCentre = (yMin + yMax)*0.5

            # If normal already in Z-direction
            if normalVec[0] == 0.00 and normalVec[1] == 0.00 and normalVec[2] != 0:
                # Given a Separate Value
                thetaAngle = 9999
                # Find a 2d point at unit distance from centre point in X direction as decided
                x2 = xCentre + cos(0)
                y2 = yCentre + sin(0)

            else:
                # Project normal vector on X-Y plane if not in Z direction
                nVec = array([0, 0, 1])
                dotProd = dot(normalVec, nVec)
                proj2dVec = array([(direction.X()-(dotProd*nVec[0])), (direction.Y()-(dotProd*nVec[1]))])  # ignore z value
                xVec = array([1, 0]) # Ignore z, as we take 2d plane
                dotProduct = xVec[0]*proj2dVec[0] + xVec[1]*proj2dVec[1]      # dot product
                det = xVec[0]*proj2dVec[1] - xVec[1]*proj2dVec[0]      # determinant
                thetaAngle = atan2(det, dotProduct)
                # Find a 2d point at unit distance from centre point in thetaAngle direction if thetaAngle not infinite(normalVec already in z-dir case)
                x2 = xCentre + cos(thetaAngle)
                y2 = yCentre + sin(thetaAngle)

            # Rotate back thetaAngle direction to the original plane
            inverseRotMatrix = inv(RotMatrix)
            if thetaAngle != 9999:
                dirVector2d = array([[cos(thetaAngle)], [sin(thetaAngle)], [0]])
            else:
                dirVector2d = array([[cos(0)], [sin(0)], [0]])
            dirVector3d = Normalize(inverseRotMatrix.dot(dirVector2d))

            # Rotate centre point back to original plane and make an edge in the thetaAngle dir
            centrePoint3d = inverseRotMatrix.dot(array([[xCentre], [yCentre], [tempPoints[2,0]]]))
            startPointOfEdge = centrePoint3d - 10000*dirVector3d
            endPointOfEdge = centrePoint3d + 10000*dirVector3d
            edge1 = BRepBuilderAPI_MakeEdge(gp_Pnt(startPointOfEdge[0,0],startPointOfEdge[1,0],startPointOfEdge[2,0]),gp_Pnt(endPointOfEdge[0,0],endPointOfEdge[1,0],endPointOfEdge[2,0])).Edge()

            # Find the edge1 intersection with edges on the current wire
            allIntersectionPoints = []
            allIntersectionPoints2d = []
            intersection2dPoints = {}
            for order in crvOrder:
                edge2 = markers[order]
                dss = BRepExtrema_DistShapeShape(edge1, edge2)
                if dss.Value() < 0.0001:
                    intersectPoints = []
                    rotated2dPoint = []
                    # Store the intersection points in intersectPointsValues and intersection2dPoints dictionary
                    for i in range(1, dss.NbSolution()+1):
                        containsPoint = False
                        # Check if the point already exists in the list
                        for apoint in allIntersectionPoints:
                            if apoint == dss.PointOnShape2(i):
                                containsPoint = True
                        if not containsPoint:
                            intersectPoints.append(dss.PointOnShape2(i))
                            allIntersectionPoints.append(dss.PointOnShape2(i))
                            # Rotate the point to the rotated X-Y plane
                            rotatedPoint = RotMatrix.dot(array([[dss.PointOnShape2(i).X()], [dss.PointOnShape2(i).Y()], [dss.PointOnShape2(i).Z()]]))
                            rotated2dPoint.append(rotatedPoint[0:2,:])
                            allIntersectionPoints2d.append(rotatedPoint[0:2,:])
                    if intersectPoints:
                        intersectPointsValues[order] = intersectPoints
                        intersection2dPoints[order] = rotated2dPoint

            # Find the correct starting point among the intersection points if intersection happens
            if intersectPointsValues:
                xOrYList = []
                if (-pi)/4 <= thetaAngle <= pi/4:
                    # Correct 2d point will be one with maximum X-Value
                    for points2d in allIntersectionPoints2d:
                        xOrYList.append(points2d[0,0])
                    correct2dPointArray = allIntersectionPoints2d[xOrYList.index(max(xOrYList))]
                elif pi/4 < thetaAngle < (3*pi)/4:
                    # Correct 2d point will be one with maximum Y-Value
                    for points2d in allIntersectionPoints2d:
                        xOrYList.append(points2d[1,0])
                    correct2dPointArray = allIntersectionPoints2d[xOrYList.index(max(xOrYList))]
                elif thetaAngle <= (-(3*pi))/4  and thetaAngle >= (3*pi)/4:
                    # Correct 2d point will be one with minimum X-Value
                    for points2d in allIntersectionPoints2d:
                        xOrYList.append(points2d[0,0])
                    correct2dPointArray = allIntersectionPoints2d[xOrYList.index(min(xOrYList))]
                elif (-(3*pi))/4 <= thetaAngle <= (-pi)/4:
                    # Correct 2d point will be one with minimum Y-Value
                    for points2d in allIntersectionPoints2d:
                        xOrYList.append(points2d[1,0])
                    correct2dPointArray = allIntersectionPoints2d[xOrYList.index(min(xOrYList))]
                elif thetaAngle == 9999:
                    # Special case for Original plane being parallel to X-Y Plane.
                    # We have decided correct 2d point will be one with minimum X-Value
                    for points2d in allIntersectionPoints2d:
                        xOrYList.append(points2d[0,0])
                    correct2dPointArray = allIntersectionPoints2d[xOrYList.index(min(xOrYList))]

                # Find correct starting point in actual plane and also find index of edge containing it
                for key in intersection2dPoints.keys():
                    pointList2d = intersection2dPoints[key]
                    for i,point2dArray in enumerate(pointList2d):
                        if (point2dArray==correct2dPointArray).all():
                            indexOfEdgeContainingFinal3dStartingPoint = key
                            final3dStartingPoint = array([[intersectPointsValues[key][i].X()], [intersectPointsValues[key][i].Y()], [intersectPointsValues[key][i].Z()]])
                            # Break Inner Loop
                            break
                    else:
                        # Continue if inner loop not broken
                        continue
                    # Break outer loop as inner loop was broken
                    break

                # Check whether correct starting point is already in the list
                points3d = edgePointDict[indexOfEdgeContainingFinal3dStartingPoint]
                nPoints = points3d.shape[1]
                IntersectionAlreadyInPointList = False
                needToSplitEdge = True
                for j in range(nPoints):
                    if (abs(points3d[0,j] - final3dStartingPoint[0,0]) < 0.0001) and (abs(points3d[1,j] - final3dStartingPoint[1,0]) < 0.0001) and (abs(points3d[2,j] - final3dStartingPoint[2,0]) < 0.0001):
                        IntersectionAlreadyInPointList = True
                        # Starting point is already in the point list. Check if it`s luckily at the start or end of edge
                        if j == 0:
                            # No need to spit edge
                            needToSplitEdge = False
                            # Reorder crvOrder accordingly
                            crvOrder = ReorderList(crvOrder, crvOrder[crvOrder.index(indexOfEdgeContainingFinal3dStartingPoint)])

                        elif j == nPoints-1 and crvOrder[len(crvOrder)-1] != indexOfEdgeContainingFinal3dStartingPoint:
                            # Next edge will be the starting edge
                            needToSplitEdge = False
                            # Reorder crvOrder accordingly
                            crvOrder = ReorderList(crvOrder, crvOrder[crvOrder.index(indexOfEdgeContainingFinal3dStartingPoint) + 1])

                        # Starting point is already in the point list with index j which is not start or finish
                        else:
                            # Split the point, normal and tangent arrays into two parts at intersection point
                            splittedPointsArrays = hsplit(edgePointDict[indexOfEdgeContainingFinal3dStartingPoint], [j+1,nPoints])
                            splittedNormalsArrays = hsplit(edgeNormalDict[indexOfEdgeContainingFinal3dStartingPoint], [j+1,nPoints])
                            splittedTangentsArrays = hsplit(edgeTangentDict[indexOfEdgeContainingFinal3dStartingPoint], [j+1,nPoints])
                            edgePointDict[indexOfEdgeContainingFinal3dStartingPoint] = splittedPointsArrays[0]
                            edgeNormalDict[indexOfEdgeContainingFinal3dStartingPoint] = splittedNormalsArrays[0]
                            edgeTangentDict[indexOfEdgeContainingFinal3dStartingPoint] = splittedTangentsArrays[0]

                            # Create a new edge and store points normals and tangents for it along with intersection point,normal and tangent added at beginning
                            edgePointDict[edgeIndex] = append(edgePointDict[indexOfEdgeContainingFinal3dStartingPoint][:,j:j+1], splittedPointsArrays[1], axis=1)
                            edgeNormalDict[edgeIndex] = append(edgeNormalDict[indexOfEdgeContainingFinal3dStartingPoint][:,j:j+1], splittedNormalsArrays[1], axis=1)
                            edgeTangentDict[edgeIndex] = append(edgeTangentDict[indexOfEdgeContainingFinal3dStartingPoint][:,j:j+1], splittedTangentsArrays[1], axis=1)

                            # Add the newly created edge to the crvOrder
                            crvOrder.insert( (crvOrder.index(indexOfEdgeContainingFinal3dStartingPoint) + 1), edgeIndex )

                            # Reorder the crvOrder so our new edge is at the beginning
                            crvOrder = ReorderList(crvOrder, edgeIndex)

                # Find the neighboring points to the correct starting point in the edge if correct starting point is not in the list
                if not IntersectionAlreadyInPointList:
                    v1 = array([points3d[0,0:(nPoints-1)]-final3dStartingPoint[0,0], points3d[1,0:(nPoints-1)]-final3dStartingPoint[1,0], points3d[2,0:(nPoints-1)]-final3dStartingPoint[2,0]])
                    v2 = array([points3d[0,1:nPoints]-final3dStartingPoint[0,0], points3d[1,1:nPoints]-final3dStartingPoint[1,0], points3d[2,1:nPoints]-final3dStartingPoint[2,0]])
                    dotArray = array([v1.transpose()[:,0]*v2.transpose()[:,0] + v1.transpose()[:,1]*v2.transpose()[:,1] + v1.transpose()[:,2]*v2.transpose()[:,2]])
                    indexOfPrevPointToStartingPoint = dotArray.argmin(axis=1)

                    # Find the face that this edge belonged to
                    face = TopoDS.topods_Face(TopoDS.TopoDS_Shape())
                    if section.HasAncestorFaceOn1(markers[indexOfEdgeContainingFinal3dStartingPoint], face):
                        surf = BRepLProp_SLProps(BRepAdaptor_Surface(face), 1, 0.000001)
                        # Extract UV Curve
                        curve2D, uMin2D, uMax2D = BRep_Tool_CurveOnSurface(markers[indexOfEdgeContainingFinal3dStartingPoint], face)
                        # Extract 3D Curve
                        curve3D, uMin, uMax = BRep_Tool_Curve(markers[indexOfEdgeContainingFinal3dStartingPoint])
                        cLen = edgePointDict[indexOfEdgeContainingFinal3dStartingPoint].shape[1]
                        # Compute Curve Points
                        uRef = linspace(uMin, uMax, cLen)
                        uRef2D = linspace(uMin2D, uMax2D, cLen)
                        k = indexOfPrevPointToStartingPoint
                        u1 = uRef[k]
                        u2 = uRef[k+1]
                        u1u2XYZDistance = sqrt((points3d[0,k+1]-points3d[0,k])**2 + (points3d[1,k+1]-points3d[1,k])**2 + (points3d[2,k+1]-points3d[2,k])**2)
                        u1ToStartPointDistance = sqrt((final3dStartingPoint[0,0]-points3d[0,k])**2 + (final3dStartingPoint[1,0]-points3d[1,k])**2 + (final3dStartingPoint[2,0]-points3d[2,k])**2)
                        uStartPoint = u1 + (u2-u1)*(u1ToStartPointDistance/u1u2XYZDistance)
                        uStartPoint2d = uRef2D[k] + (uRef2D[k+1]-uRef2D[k])*(u1ToStartPointDistance/u1u2XYZDistance)
                        # Eval Pnt and Tangent
                        curve3D.D1(uStartPoint[0], evalPnt, evalTan)
                        # Eval Normal
                        curve2D.D0(uStartPoint2d[0], evalPnt2d)
                        surf.SetParameters(evalPnt2d.X(), evalPnt2d.Y())
                        evalNor = surf.Normal()
                        if face.Orientation() == TopAbs_REVERSED:
                            evalNor.Reverse()
                        loopStartPoint3d = array([[evalPnt.X()], [evalPnt.Y()], [evalPnt.Z()]])
                        loopStartPoint3dTan = array([[evalTan.X()], [evalTan.Y()], [evalTan.Z()]])
                        loopStartPoint3dNor = array([[evalNor.X()], [evalNor.Y()], [evalNor.Z()]])

                    # Split the point, normal and tangent arrays into two parts at intersection point
                    splittedPointsArrays = hsplit(edgePointDict[indexOfEdgeContainingFinal3dStartingPoint], [indexOfPrevPointToStartingPoint+1,nPoints])
                    splittedNormalsArrays = hsplit(edgeNormalDict[indexOfEdgeContainingFinal3dStartingPoint], [indexOfPrevPointToStartingPoint+1,nPoints])
                    splittedTangentsArrays = hsplit(edgeTangentDict[indexOfEdgeContainingFinal3dStartingPoint], [indexOfPrevPointToStartingPoint+1,nPoints])
                    edgePointDict[indexOfEdgeContainingFinal3dStartingPoint] = append(splittedPointsArrays[0], final3dStartingPoint, axis=1)
                    edgeNormalDict[indexOfEdgeContainingFinal3dStartingPoint] = append(splittedNormalsArrays[0], loopStartPoint3dNor, axis=1)
                    edgeTangentDict[indexOfEdgeContainingFinal3dStartingPoint] = append(splittedTangentsArrays[0], loopStartPoint3dTan, axis=1)

                    # Create a new edge and store points normals and tangents for it along with intersection point,normal and tangent added at beginning
                    edgePointDict[edgeIndex] = append(final3dStartingPoint, splittedPointsArrays[1], axis=1)
                    edgeNormalDict[edgeIndex] = append(loopStartPoint3dNor, splittedNormalsArrays[1], axis=1)
                    edgeTangentDict[edgeIndex] = append(loopStartPoint3dTan, splittedTangentsArrays[1], axis=1)

                    # Add the newly created edge to the crvOrder
                    crvOrder.insert( (crvOrder.index(indexOfEdgeContainingFinal3dStartingPoint) + 1), edgeIndex )

                    # Reorder the crvOrder so our new edge is at the beginning
                    crvOrder = ReorderList(crvOrder, edgeIndex)
        # Finding correct starting point and splitting the edge containing it and reordering crvOrder ends


        # Check orientation for open loop:
        if not wire.Closed() and crvOrder:
            normalVec = array([direction.X(), direction.Y(), direction.Z()])

            # If normal of original plane already in Z-direction, set a value for thetaAngle to be identifiable
            if normalVec[0] == 0.00 and normalVec[1] == 0.00 and normalVec[2] != 0.00:
                # Given a separate value
                thetaAngle = 9999

            else:
                # Project normal vector on X-Y plane (For else case) and find it's angle with positive X-Axis
                nVec = array([0, 0, 1])
                dotProd = dot(normalVec, nVec)
                proj2dVec = array([(direction.X()-(dotProd*nVec[0])), (direction.Y()-(dotProd*nVec[1]))])  # ignore z value
                xVec = array([1, 0]) # Ignore z, as we take 2d plane
                dotProduct = xVec[0]*proj2dVec[0] + xVec[1]*proj2dVec[1]      # dot product
                det = xVec[0]*proj2dVec[1] - xVec[1]*proj2dVec[0]      # determinant
                thetaAngle = atan2(det, dotProduct)

            openLoopFlip = False
            openLoopStartPoint = edgePointDict[crvOrder[0]][:,0]
            openLoopEndPoint = edgePointDict[crvOrder[len(crvOrder)-1]][:,edgePointDict[crvOrder[len(crvOrder)-1]].shape[1]-1]
            if (-pi)/4 <= thetaAngle <= pi/4:
                # Correct 2d point will be one with maximum X-Value
                if openLoopStartPoint[1] < openLoopEndPoint[1]:
                    openLoopFlip = True
            elif pi/4 < thetaAngle < (3*pi)/4:
                # Correct 2d point will be one with maximum Y-Value
                if openLoopEndPoint[0] < openLoopStartPoint[0]:
                    openLoopFlip = True
            elif thetaAngle <= (-(3*pi))/4  and thetaAngle >= (3*pi)/4:
                # Correct 2d point will be one with minimum X-Value
                if openLoopEndPoint[1] < openLoopStartPoint[1]:
                    openLoopFlip = True
            elif (-(3*pi))/4 <= thetaAngle <= (-pi)/4:
                # Correct 2d point will be one with minimum Y-Value
                if openLoopStartPoint[0] < openLoopEndPoint[0]:
                    openLoopFlip = True
            elif thetaAngle == 9999:
                # Correct 2d point will be one with minimum X-Value as decided if original plane is parallel to X-Y plane
                if openLoopEndPoint[1] < openLoopStartPoint[1]:
                    openLoopFlip = True

            # Flip the points, normals and tangents on the loop
            if not openLoopFlip:
                crvOrder = crvOrder[::-1]
                for correctIndex in crvOrder:
                    edgePointDict[correctIndex] = fliplr(edgePointDict[correctIndex])
                    edgeNormalDict[correctIndex] = fliplr(edgeNormalDict[correctIndex])
                    edgeTangentDict[correctIndex] = -fliplr(edgeTangentDict[correctIndex])


        # Add the points to the final points array in correct order
        for correctIndex in crvOrder:
            points = append(points, edgePointDict[correctIndex], axis=1)
            normals = append(normals, edgeNormalDict[correctIndex], axis=1)
            tangents = append(tangents, edgeTangentDict[correctIndex], axis=1)
            
    #print time.time()-start
    return points, normals, tangents            

def meshSectioning(aMesh, cutter, direction, sampling):

    """!
    Performs sectioning on a mesh object along a given direction.
    @param aMesh: Mesh object
    @param cutter:
    @param direction:
    @param sampling:
    @return: tuple of points list. normals list, tangents list, markers list
    """

    if str(aMesh.__class__.__name__) == "Mesh":
        points = array([[],[],[]]);
        normals = array([[],[],[]]);
        tangents = array([[],[],[]]);
        markers = []
        edgeMarkers = []
        edgeList = TopTools_HSequenceOfShape()
        from OCC.CTM import DoubleVector3List, Vector3
        sectionNormals = DoubleVector3List()
        try:
            from OCC.CTM import OCCCTM_Section
            successSection = OCCCTM_Section(aMesh, cutter, edgeList, sectionNormals)
        except:
            print("Sectioning at Plane Location " , cutter.Location(), " failed.")
            return points, normals, tangents, markers

        # Attach normal with the edges
        sectionEdgeNormalList = DoubleVector3List()
        for edgeIndex in xrange(1,edgeList.Length()+1):
            anedge = TopoDS.topods_Edge(edgeList.Value(edgeIndex))
            acurve = BRepAdaptor_Curve(anedge)
            aStartPoint = acurve.Value(acurve.FirstParameter())
            anEndPoint = acurve.Value(acurve.LastParameter())
            sectionEdgeNormalList.push_back(Vector3(aStartPoint.X(), aStartPoint.Y(), aStartPoint.Z()))
            sectionEdgeNormalList.push_back(Vector3(anEndPoint.X(), anEndPoint.Y(), anEndPoint.Z()))
            sectionEdgeNormalList.push_back(Vector3(sectionNormals[edgeIndex-1].X(), sectionNormals[edgeIndex-1].Y(), sectionNormals[edgeIndex-1].Z()))

        # List of multiple wires generated by ConnectEdgesToWires algorithm
        wireList = TopTools_HSequenceOfShape()
        wireListHandle = wireList
        ShapeAnalysis_FreeBounds_ConnectEdgesToWires(edgeList, 1e-3, False, wireListHandle)

        #Iterating through each wire and then through the edges in each wire
        wires = []
        for wireIndex in xrange(1,wireListHandle.Length()+1):
            wire = wireListHandle.Value(wireIndex)
            # Wire Healing Tool
            sewd = ShapeExtend_WireData()
            sfw = ShapeFix_Wire()
            exp = BRepTools_WireExplorer(TopoDS.topods_Wire(wire))
            while exp.More():
                edge = exp.Current()
                sewd.Add(edge)
                sfw.Load(sewd)
                sfw.Perform()
                sfw.FixReorder()
                sfw.SetMaxTolerance(0.01)
                sfw.SetClosedWireMode(True)
                sfw.FixConnected(0.01)
                sfw.FixClosed(0.01)
                exp.Next()
            wires.append(sfw.Wire())

        # Connect Wires to Wire (Test)
        openWireList = TopTools_HSequenceOfShape()
        closedWireList = TopTools_HSequenceOfShape()
        newWires = []
        if wireListHandle.Length() > 1:
            for wire in wires:
                if not wire.Closed():
                    openWireList.Append(wire)
                else:
                    newWires.append(wire)
        else:
            for wire in wires:
                newWires.append(wire)
        tol = 0.001
        while wireListHandle.Length() > 1:
            wireList = TopTools_HSequenceOfShape()
            wireListHandle = wireList
            ShapeAnalysis_FreeBounds_ConnectWiresToWires(openWireList, tol, False, wireListHandle)
            tol = tol + 0.001
        if openWireList.Length() > 0 and wireListHandle.Length() == 1:
            newWires.append(wireListHandle.Value(1))
        # Test Work Done

        if len(newWires) == 0:
            return points, normals, tangents, markers

        # # Initialize OCC vars
        evalPnt = gp_Pnt()
        evalPnt2d = gp_Pnt2d()
        evalTan = gp_Vec()

        for wire in newWires:
            markers.append(wire)
            # Temp arrays for storing points
            pointsBeforeReorder = array([[],[],[]]);
            normalsBeforeReorder = array([[],[],[]]);
            tangentsBeforeReorder = array([[],[],[]]);

            # Dictionary for storing crvPoints, crvNormals, crvTangents for correcting their order later
            edgeIndex = 0
            edgePointDict = {}
            edgeNormalDict = {}
            edgeTangentDict = {}
            # Explore wire (to get edges)
            exp = BRepTools_WireExplorer(TopoDS.topods_Wire(wire))
            while exp.More():
                edge = exp.Current()
                curve3D, uMin, uMax = BRep_Tool_Curve(edge)
                # Get Linear Properties
                props = GProp_GProps()
                brepgprop_LinearProperties(edge, props)
                length = props.Mass()
                # Initialize Numpy Curves
                cLen = max(int(ceil(length/1)), 3)
                crvPoints = zeros((3,cLen))
                crvNormals = zeros((3,cLen))
                crvTangents = zeros((3,cLen))
                crvU = zeros(cLen); crvV = zeros(cLen)
                # --- Compute Curve Points ---
                uRef = linspace(uMin, uMax, cLen)

                #Normal Calculation
                startPnt = gp_Pnt()
                endPnt = gp_Pnt()
                curve3D.D0(uMin, startPnt)
                curve3D.D0(uMax, endPnt)
                normalVecForEdge = Vector3()
                from OCC.CTM import OCCCTM_GetNormalForMeshEdge
                gotNormal = OCCCTM_GetNormalForMeshEdge(Vector3(startPnt.X(), startPnt.Y(), startPnt.Z()), Vector3(endPnt.X(), endPnt.Y(), endPnt.Z()), sectionEdgeNormalList, normalVecForEdge)
                #Normal Calculation Ends

                for u in range(cLen):
                    # Eval Point and Tangent
                    curve3D.D1(uRef[u], evalPnt, evalTan)
                    # Find U/V Coordinates on Surface
                    crvU[u] = evalPnt2d.X()
                    crvV[u] = evalPnt2d.Y()
                    # Save in NumPy format
                    crvPoints[0,u] = evalPnt.X()
                    crvPoints[1,u] = evalPnt.Y()
                    crvPoints[2,u] = evalPnt.Z()
                    #magnitude = normalVecForEdge.Abs()
                    crvNormals[0,u] = normalVecForEdge.X()
                    crvNormals[1,u] = normalVecForEdge.Y()
                    crvNormals[2,u] = normalVecForEdge.Z()
                    magnitude = sqrt(evalTan.X()**2 + evalTan.Y()**2 +  evalTan.Z()**2)
                    crvTangents[0,u] = evalTan.X() / magnitude
                    crvTangents[1,u] = evalTan.Y() / magnitude
                    crvTangents[2,u] = evalTan.Z() / magnitude
                # --- Check Tangent Direction ---
                crvDir = crvPoints[:,1:cLen] - crvPoints[:,0:cLen-1]
                crvDir = append(crvDir, crvDir[:,cLen-2:cLen-1], axis=1)
                for n in range(cLen):
                    if dot(crvDir[:,n], crvTangents[:,n]) < 0:
                        crvTangents[:,n] = -crvTangents[:,n]
                # Create dictionary for points, normals, tangents and edges
                edgePointDict[edgeIndex] = crvPoints
                edgeNormalDict[edgeIndex] = crvNormals
                edgeTangentDict[edgeIndex] = crvTangents
                edgeIndex = edgeIndex + 1
                # Save edge in edgeMarkers (for display)
                edgeMarkers.append(edge)
                exp.Next()

            # The original edge/curve order given by BRep_WireExplorer
            origCrvOrder = []
            for i in range(edgeIndex):
                origCrvOrder.append(i)

            # Check edges for: order, flip them if required
            crvOrder = []
            tol = 0.0001
            while len(crvOrder) < (edgeIndex):
                if len(crvOrder) == 0:
                    crvOrder.append(0)
                    if edgeIndex > 1:
                        if (dot(edgePointDict[0][:,0]-edgePointDict[1][:,0],edgePointDict[0][:,0]-edgePointDict[1][:,0]) < 0.001) or \
                                (dot(edgePointDict[0][:,0]-edgePointDict[1][:,edgePointDict[1].shape[1]-1],edgePointDict[0][:,0]-edgePointDict[1][:,edgePointDict[1].shape[1]-1]) < 0.001):
                            edgePointDict[0] = fliplr(edgePointDict[0])
                            edgeNormalDict[0] = fliplr(edgeNormalDict[0])
                            edgeTangentDict[0] = -fliplr(edgeTangentDict[0])
                    currentCurve = edgePointDict[0]
                    currCurveLen = edgePointDict[0].shape[1]
                else:
                    checkLength = len(crvOrder)
                    for i in range(edgeIndex):
                        if i in crvOrder:
                            continue
                        cLen = edgePointDict[i].shape[1]
                        if (abs(edgePointDict[i][0,0] - currentCurve[0,currCurveLen-1]) < tol) and \
                                (abs(edgePointDict[i][1,0] - currentCurve[1,currCurveLen-1]) < tol) and \
                                (abs(edgePointDict[i][2,0] - currentCurve[2,currCurveLen-1]) < tol):
                            crvOrder.append(i)
                            currentCurve = edgePointDict[i]
                            currCurveLen = edgePointDict[i].shape[1]
                            break
                        elif (abs(edgePointDict[i][0,cLen-1] - currentCurve[0,currCurveLen-1]) < tol) and \
                                (abs(edgePointDict[i][1,cLen-1] - currentCurve[1,currCurveLen-1]) < tol) and \
                                 (abs(edgePointDict[i][2,cLen-1] - currentCurve[2,currCurveLen-1]) < tol):
                            crvOrder.append(i)
                            edgePointDict[i] = fliplr(edgePointDict[i])
                            edgeNormalDict[i] = fliplr(edgeNormalDict[i])
                            edgeTangentDict[i] = -fliplr(edgeTangentDict[i])
                            currentCurve = edgePointDict[i]
                            currCurveLen = edgePointDict[i].shape[1]
                            break
                    if checkLength == len(crvOrder):
                        tol = tol + 0.001
                    else:
                        tol = 0.0001

            # Check CCW orientation (for closed loops)
            if wire.Closed():
                # Find the longest edge (edge with maximum points)
                edgeLengthList = []
                for m in range(edgeIndex):
                    edgeLengthList.append(edgePointDict[m].shape[1])
                if not edgeLengthList:
                    continue
                maxValue = max(edgeLengthList)
                maxIndex = edgeLengthList.index(maxValue)

                # Find mid-point of longest edge
                copiedEdgePointDict = deepcopy(edgePointDict)
                copiedEdgeTangentDict = deepcopy(edgeTangentDict)
                midIndexInCrvPoints = maxValue//2
                crvPointsLongestEdge = deepcopy(copiedEdgePointDict[maxIndex])
                midPointOnCrvPoints = crvPointsLongestEdge[:,midIndexInCrvPoints]

                # Compute cross product of direction and tangent, then project from mid-point
                crvTangentLongestEdge = deepcopy(copiedEdgeTangentDict[maxIndex])
                a = array([direction.X(), direction.Y(), direction.Z()])
                b = crvTangentLongestEdge[:,midIndexInCrvPoints]
                crossProduct = cross(a,b)
                # InPointToTest = midPointOnCrvPoints + 0.01*crossProduct
                InPointToTest = midPointOnCrvPoints + 0.4*crossProduct

                # Find Rotation Matrix for rotating plane to align it with xy Plane
                tempPoints = array([[],[],[]]);
                if a[0] == 0.00 and a[1] == 0.00 and a[2] != 0:
                    # Store all the points in tempPoints
                    for correctIndex in crvOrder:
                        tempPoints = append(tempPoints, edgePointDict[correctIndex], axis=1)
                    # Delete the Z-coordinate to get 2-d planar points
                    temp2dPoints = delete(tempPoints, (2), axis=0)

                    # Set RotMatrix as Identity
                    RotMatrix = array([[ 1,  0,  0 ],
                                       [ 0,  1,  0 ],
                                       [ 0,  0,  1 ]])
                else:
                    # Unit Vector in Z-dir
                    b = array([0,0,1])
                    # Get rotation axis
                    axis = cross(b,Normalize(a))
                    # Get rotation angle
                    theta = acos((dot(a,b))/(linalg.norm(a)*linalg.norm(b)))
                    # Get rotation matrix
                    RotMatrix = RotationAxis(axis, theta)

                    # Get rotated points on the new plane parallel to x-y plane
                    for i, correctIndex in enumerate(crvOrder):
                        correctIndex = abs(correctIndex)
                        anArray = array([[],[],[]]);
                        for j in range(edgePointDict[correctIndex].shape[1]):
                            anArray = append(anArray, array([[RotMatrix.dot(copiedEdgePointDict[correctIndex][:,j])[0]],[RotMatrix.dot(copiedEdgePointDict[correctIndex][:,j])[1]],[RotMatrix.dot(copiedEdgePointDict[correctIndex][:,j])[2]]]), axis=1)
                        # Store points in tempPoints for Path method of matplotlib to check whether the InPointToTest lies inside or outside the loop
                        tempPoints = append(tempPoints, anArray, axis=1)

                    # Remove the Z-coordinate from them as they are planar points
                    temp2dPoints = delete(tempPoints, (2), axis=0)

                    # Rotate the InPointToTest on the new plane
                    InPointToTest = RotMatrix.dot(InPointToTest)

                # Check if point lies inside or outside the path
                InPointToTest = array([[InPointToTest[0]],[InPointToTest[1]]])
                p = Path(temp2dPoints.transpose())
                if p.contains_point(InPointToTest) != 1:
                    # Change orientation (Flip points, normals and tangents)
                    crvOrder = crvOrder[::-1]
                    for index in crvOrder:
                        edgePointDict[index] = fliplr(edgePointDict[index])
                        edgeNormalDict[index] = fliplr(edgeNormalDict[index])
                        edgeTangentDict[index] = -fliplr(edgeTangentDict[index])
                    # Flip the tempPoints and 2d points
                    tempPoints = fliplr(tempPoints)
                    temp2dPoints = fliplr(temp2dPoints)

            # Check orientation for open loop:
            if not wire.Closed() and crvOrder:
                normalVec = array([direction.X(), direction.Y(), direction.Z()])

                # If normal of original plane already in Z-direction, set a value for thetaAngle to be identifiable
                if normalVec[0] == 0.00 and normalVec[1] == 0.00 and normalVec[2] != 0.00:
                    # Given a separate value
                    thetaAngle = 9999

                else:
                    # Project normal vector on X-Y plane (For else case) and find it's angle with positive X-Axis
                    nVec = array([0, 0, 1])
                    dotProd = dot(normalVec, nVec)
                    proj2dVec = array([(direction.X()-(dotProd*nVec[0])), (direction.Y()-(dotProd*nVec[1]))])  # ignore z value
                    xVec = array([1, 0]) # Ignore z, as we take 2d plane
                    dotProduct = xVec[0]*proj2dVec[0] + xVec[1]*proj2dVec[1]      # dot product
                    det = xVec[0]*proj2dVec[1] - xVec[1]*proj2dVec[0]      # determinant
                    thetaAngle = atan2(det, dotProduct)

                openLoopFlip = False
                openLoopStartPoint = edgePointDict[crvOrder[0]][:,0]
                openLoopEndPoint = edgePointDict[crvOrder[len(crvOrder)-1]][:,edgePointDict[crvOrder[len(crvOrder)-1]].shape[1]-1]
                if (-pi)/4 <= thetaAngle <= pi/4:
                    # Correct 2d point will be one with maximum X-Value
                    if openLoopStartPoint[1] < openLoopEndPoint[1]:
                        openLoopFlip = True
                elif pi/4 < thetaAngle < (3*pi)/4:
                    # Correct 2d point will be one with maximum Y-Value
                    if openLoopEndPoint[0] < openLoopStartPoint[0]:
                        openLoopFlip = True
                elif thetaAngle <= (-(3*pi))/4  and thetaAngle >= (3*pi)/4:
                    # Correct 2d point will be one with minimum X-Value
                    if openLoopEndPoint[1] < openLoopStartPoint[1]:
                        openLoopFlip = True
                elif (-(3*pi))/4 <= thetaAngle <= (-pi)/4:
                    # Correct 2d point will be one with minimum Y-Value
                    if openLoopStartPoint[0] < openLoopEndPoint[0]:
                        openLoopFlip = True
                elif thetaAngle == 9999:
                    # Correct 2d point will be one with minimum X-Value as decided if original plane is parallel to X-Y plane
                    if openLoopEndPoint[1] < openLoopStartPoint[1]:
                        openLoopFlip = True

                # Flip the points, normals and tangents on the loop
                if openLoopFlip:
                    crvOrder = crvOrder[::-1]
                    for correctIndex in crvOrder:
                        edgePointDict[correctIndex] = fliplr(edgePointDict[correctIndex])
                        edgeNormalDict[correctIndex] = fliplr(edgeNormalDict[correctIndex])
                        edgeTangentDict[correctIndex] = -fliplr(edgeTangentDict[correctIndex])

            # Add the points to the final points array in correct order
            #for correctIndex in origCrvOrder:
            for correctIndex in crvOrder:
                points = append(points, edgePointDict[correctIndex], axis=1)
                normals = append(normals, edgeNormalDict[correctIndex], axis=1)
                tangents = append(tangents, edgeTangentDict[correctIndex], axis=1)

        # Find nans (Samyak tried to kill me!!!!!!)
        notNans = sum(isnan(points) | isnan(normals) | isnan(tangents), 0) == 0
        points = points[:, notNans]
        normals = normals[:, notNans]
        tangents = tangents[:, notNans]

        dist = CumDistance(points)
        pLen = ceil(dist[-1]/sampling)
        newdist = linspace(0,dist[-1],pLen)

        x = interp(newdist,dist,points[0])
        y = interp(newdist,dist,points[1])
        z = interp(newdist,dist,points[2])
        points = array([x,y,z])

        nx = interp(newdist,dist,normals[0])
        ny = interp(newdist,dist,normals[1])
        nz = interp(newdist,dist,normals[2])
        normals = array([nx,ny,nz])

        tx = interp(newdist,dist,tangents[0])
        ty = interp(newdist,dist,tangents[1])
        tz = interp(newdist,dist,tangents[2])
        # Filter for removing spurious tangents generated at the edge of sections (not fully tested).
        mt = mean([tx, ty, tz], 1)
        for ind in range(len(tx)):
            if dot(mt, array([tx[ind], ty[ind], tz[ind]])) < 0:
                tx[ind] = -tx[ind]
                ty[ind] = -ty[ind]
                tz[ind] = -tz[ind]
        tangents = array([tx,ty,tz])

        return points, normals, tangents, markers

def ReorderList(alist, aValue):
    """!Sort list
    @param alist: list to re-order
    @param aValue: value in the list
    @return: tuple of points list, normals list and tangents list
    """
    tempOrder1 = []
    tempOrder2 = []
    temp = alist.index(aValue)
    for i,order in enumerate(alist):
        if i < temp:
            tempOrder2.append(order)
        else:
            tempOrder1.append(order)
    alist = []
    for order in tempOrder1:
        alist.append(order)
    for order in tempOrder2:
        alist.append(order)
    return alist            

    #print time.time()-start
    return points, normals, tangents

#### Edge/Wire methods ####

def EdgesToWires(edges):
    """!
    Make all possible wires from edges
    @param edges: edges
    """

    # List for collecting all edges from section result
    edgeList = TopTools_HSequenceOfShape()
    for edge in edges:
        edgeList.Append(edge)

    # List of multiple wires to be generated by ConnectEdgesToWires algorithm
    wireList = TopTools_HSequenceOfShape()
    wireListHandle = wireList.GetHandle()
    ShapeAnalysis_FreeBounds_ConnectEdgesToWires(edgeList.GetHandle(), 1e-3, False, wireListHandle)

    # Iterating through each wire and then through the edges in each wire
    aWireList = TopTools_HSequenceOfShape()
    for wireIndex in xrange(1, wireListHandle.GetObject().Length() + 1):
        wire = wireListHandle.GetObject().Value(wireIndex)
        # Wire Healing Tool
        sewd = ShapeExtend_WireData()
        sfw = ShapeFix_Wire()
        exp = BRepTools_WireExplorer(TopoDS.topods_Wire(wire))
        while exp.More():
            edge = exp.Current()
            sewd.Add(edge)
            sfw.Load(sewd.GetHandle())
            sfw.Perform()
            sfw.FixReorder()
            sfw.SetMaxTolerance(0.01)
            sfw.SetClosedWireMode(True)
            sfw.FixConnected(0.01)
            sfw.FixClosed(0.01)
            exp.Next()
        aWireList.Append(sfw.Wire())

    wireList = TopTools_HSequenceOfShape()
    wireListHandle = wireList.GetHandle()
    ShapeAnalysis_FreeBounds_ConnectWiresToWires(aWireList.GetHandle(), 1e-3, False, wireListHandle)

    # Iterating through each wire and fixing it
    finalWires = []
    for wireIndex in xrange(1, wireListHandle.GetObject().Length() + 1):
        wire = wireListHandle.GetObject().Value(wireIndex)
        # Wire Healing Tool
        sewd = ShapeExtend_WireData()
        sfw = ShapeFix_Wire()
        exp = BRepTools_WireExplorer(TopoDS.topods_Wire(wire))
        while exp.More():
            edge = exp.Current()
            sewd.Add(edge)
            sfw.Load(sewd.GetHandle())
            sfw.Perform()
            sfw.FixReorder()
            sfw.SetMaxTolerance(0.01)
            sfw.SetClosedWireMode(True)
            sfw.FixConnected(0.01)
            sfw.FixClosed(0.01)
            exp.Next()
        # display.DisplayColoredShape(sfw.Wire(), 'BLACK')
        finalWires.append(sfw.Wire())
    return finalWires


# Sample points on wire
def SampleWire(wire, sampling=0.1):
    """!
    Sample points on wire
    @param wire: wire object to sample
    @param sampling: sampling resolution
    @return: array of points
    """

    # Initialize point array
    points = array([[], [], []])
    for edge in Topo(wire).edges():
        # Get the underlying curve
        curveHandle, uMin, uMax = BRep_Tool_Curve(edge)
        curve = curveHandle.GetObject()

        # Get Linear Properties
        props = GProp_GProps()
        brepgprop_LinearProperties(edge, props)
        length = props.Mass()
        # Safety length check
        if length < 0.01:
            continue

        # Sample Points
        cLen = max(int(ceil(length / sampling)), 3)
        uRef = linspace(uMin, uMax, cLen)
        for u in uRef:
            # Save points in NumPy format
            evalPnt = curve.Value(u)
            points = concatenate((points, [[evalPnt.X()], [evalPnt.Y()], [evalPnt.Z()]]), axis=1)
    return points


def SampleWireUV(wire, face, sampling=0.1):
    surfHandle = BRep_Tool_Surface(face)
    sas = ShapeAnalysis_Surface(surfHandle)

    # Go through the edges of the wire to get the points for in-polygon
    pointsUV = array([[], []])
    points = SampleWire(wire, sampling=sampling)
    for i in range(len(points[0])):
        uvVal = sas.ValueOfUV(gp_Pnt(points[0][i], points[1][i], points[2][i]), 0.01)
        pointsUV = concatenate((pointsUV, [[uvVal.X()], [uvVal.Y()]]), axis=1)
    return pointsUV


def HealWire(wire):
    """!
    Wire Healing Tool
    @param wire: wire object to heal
    """
    sewd = ShapeExtend_WireData()
    sfw = ShapeFix_Wire()
    exp = BRepTools_WireExplorer(TopoDS.topods_Wire(wire))
    while exp.More():
        edge = exp.Current()
        sewd.Add(edge);
        sfw.Load(sewd.GetHandle())
        sfw.Perform();
        sfw.FixReorder()
        sfw.SetMaxTolerance(0.1);
        sfw.SetClosedWireMode(True)
        sfw.FixConnected(0.1);
        sfw.FixClosed(0.1)
        exp.Next()
    return sfw.Wire()


# Get wire from intersection of face and cutter
def GetSectionWire(face, cutter, solid=None):
    """!
    Get wire from intersection of face and cutter
    @param face: face object
    @param cutter: ?
    @param solid: ?
    @return:
    """
    section = BRepAlgoAPI_Section(face, cutter)

    # Get edges from sestion shape
    edges = []
    for edge in Topo(section.Shape()).edges():
        edges.append(edge)

    # Check if the section is already a single wire and is closed
    wire = None
    if len(edges) == 1 and edges[0].Closed():
        # Special case
        wire = BRepBuilderAPI_MakeWire(edges[0]).Wire()
    else:
        ##### General case #####
        ### Algorith using wires to split shape - make all possible wires from the edges. Then make cases:
        ### if 1 wire: split shape and get the correct part, If more than 1 wire,
        ### split in progression keeping the correct result from every splitting
        wires = EdgesToWires(edges)
        if len(wires) == 1:
            if wires[0].Closed():
                wire = wires[0]
            else:
                wire = wires[0]
                wire = HealWire(wire)
                # Project it on the face normally
                brnp = BRepAlgo_NormalProjection(face)
                brnp.Add(wire)
                brnp.Build()
                aWireList = TopTools_ListOfShape()
                brnp.BuildWire(aWireList)
                wire = TopoDS.topods_Wire(aWireList.First())
                # Case of only one open wire on the face
                outShapes = []
                splitter = BRepFeat_SplitShape(face)
                splitter.Add(TopoDS.topods_Wire(wire), face)
                splitter.Build()
                if splitter.IsDone():
                    iter = TopTools_ListIteratorOfListOfShape(splitter.Modified(face))
                    # iter = TopTools_ListIteratorOfListOfShape(splitter.Left())
                    while iter.More():
                        outShapes.append(iter.Value())
                        iter.Next()
                splitter.Delete()
                # Get the correct one from the list of resultant shapes
                correctFace = TopoDS.topods_Face(outShapes[0])
                if solid:
                    solidClassifier = BRepClass3d_SolidClassifier(solid)
                    correctFace = TopoDS.topods_Face(outShapes[0])
                    for aShape in outShapes:
                        aFace = TopoDS.topods_Face(aShape)
                        uMin, uMax, vMin, vMax = breptools_UVBounds(aFace)
                        surface = BRepAdaptor_Surface(aFace)
                        point = gp_Pnt()
                        surface.D0((uMin + uMax) / 2.0, (vMin + vMax) / 2.0, point)
                        # Check if point is inside the solid or not
                        solidClassifier.Perform(point, 0.001)
                        if solidClassifier.State() != TopAbs_OUT:
                            correctFace = aFace
                            break
                wire = breptools_OuterWire(correctFace)

        elif len(wires) == 0:
            # Go through all edges
            pass
        else:
            finalFace = face
            for wInd, aWire in enumerate(wires):
                aWire = HealWire(aWire)
                # Project it on the face normally
                brnp = BRepAlgo_NormalProjection(face)
                brnp.Add(aWire)
                brnp.Build()
                aWireList = TopTools_ListOfShape()
                brnp.BuildWire(aWireList)
                aWire = TopoDS.topods_Wire(aWireList.First())
                outShapes = []
                splitter = BRepFeat_SplitShape(face)
                splitter.Add(TopoDS.topods_Wire(aWire), face)
                splitter.Build()
                if splitter.IsDone():
                    iter = TopTools_ListIteratorOfListOfShape(splitter.Modified(face))
                    while iter.More():
                        outShapes.append(iter.Value())
                        iter.Next()
                splitter.Delete()
                if wInd == 0:
                    otherWire = wires[1]
                else:
                    otherWire = wires[0]
                finalFace = outShapes[0]
                for aShape in outShapes:
                    dss = BRepExtrema_DistShapeShape(aShape, otherWire)
                    if dss.IsDone() and dss.Value() < 0.01:
                        finalFace = TopoDS.topods_Face(aShape)
                        break
            # Get the final wire which is the outer wire of the final face
            wire = breptools_OuterWire(face)

        ### Other algo which uses edge edge intersection
        # topEI = TopOpeBRep_EdgesIntersector()
        # topEI.SetFaces(face, face)

    return wire

def WiresFromSectioning(section):
    # List for collecting all edges from section result
    edgeList = TopTools_HSequenceOfShape()
    newEdgesList = []
    for edge in Topo(section.Shape()).edges():
        # Check for edge length and continue if too small
        props = GProp_GProps()
        brepgprop_LinearProperties(edge, props)
        length = props.Mass()
        if length < 0.1:  # Length less than 10 um, ignore
            continue
        edgeList.Append(edge)
        newEdgesList.append(edge)

    # List of multiple wires generated by ConnectEdgesToWires algorithm
    wireListHandle = ShapeAnalysis_FreeBounds_ConnectEdgesToWiresCustom(edgeList, 1e-2, False)

    if wireListHandle.Length() > 0:
        # Iterating through each wire and then through the edges in each wire
        wires = []
        for wireIndex in range(1, wireListHandle.Length() + 1):
            wire = wireListHandle.Value(wireIndex)

            # Wire Healing Tool
            sewd = ShapeExtend_WireData()
            sfw = ShapeFix_Wire()
            exp = BRepTools_WireExplorer(TopoDS.topods_Wire(wire))
            while exp.More():
                edge = exp.Current()
                sewd.Add(edge)
                sfw.Load(sewd)
                sfw.Perform()
                sfw.FixReorder()
                sfw.SetMaxTolerance(0.01)
                sfw.SetClosedWireMode(True)
                sfw.FixConnected(0.01)
                sfw.FixClosed(0.01)
                exp.Next()
            wires.append(sfw.Wire())

        # Connect Wires to Wire (Test)
        openWireList = TopTools_HSequenceOfShape()
        # closedWireList = TopTools_HSequenceOfShape()
        newWires = []
        if wireListHandle.Length() > 1:
            for wire in wires:
                if not wire.Closed():
                    openWireList.Append(wire)
                else:
                    newWires.append(wire)
        else:
            for wire in wires:
                newWires.append(wire)
        tol = 0.005
        while wireListHandle.Length() > 1:
            wireListHandle = ShapeAnalysis_FreeBounds_ConnectWiresToWiresCustom(openWireList, tol, False)
            tol = tol * 2
        if openWireList.Length() > 1 and wireListHandle.Length() == 1:
            newWires.append(wireListHandle.Value(1))
            # Test Work Done

        return newWires
    else:
        if len(newEdgesList) == 0:
            return []

        wireList = BuildWiresFromEdgeset(newEdgesList)
        return wireList

    
###############################################################################

from OCC.Mesh.Mesh import QuickTriangleMesh

def cad2mesh(shape, quality=1.):
    """!
    Convert shape to mesh?
    @param shape: shape object
    @param quality: float
    @return: tuple containing a list of vertices and a list of triangles/faces
    """
    mesh = QuickTriangleMesh()
    mesh.set_shape(shape)

    # define precision for the mesh
    mesh.set_precision(mesh.get_precision() / (quality * 10.))

    # then compute the mesh
    mesh.compute()

    # Extract vertices/faces            
    v = mesh.get_vertices()
    f = [];
    i = 0
    while i < len(v):
        f.append([i, i + 1, i + 2])
        i = i + 3

    # Convert to numpy format
    vertices = transpose(array(v))
    triangles = array(f)

    # Output data
    return vertices, triangles            

def cadbounds(shape):
    """!
    Compute bounds using mesh
    @param shape: object
    @return: boundary data
    """
    vertices, triangles = cad2mesh(shape)
    return min(vertices[0]), min(vertices[1]), min(vertices[2]), max(vertices[0]), max(vertices[1]), max(vertices[2])

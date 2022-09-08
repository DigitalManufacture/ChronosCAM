__author__ = "Anthony Beaucamp"
__copyright__ = "Copyright (C) 2022 Kyoto University"
__license__ = "Public Domain"
__version__ = "1.0"

from math import *
from numpy import *

from OCC.Core.gp import *
from OCC.Core.BRepAlgoAPI import *
from OCC.Core.BRepBuilderAPI import *
from OCC.Core.BRepPrimAPI import *
from OCC.Core.GeomAbs import *
from OCC.Core.GeomAPI import *
from OCC.Core.TColgp import *
from OCC.Core.TopLoc import *
from OCC.Core.TopoDS import *

def GenerateRuledGeometry(occ):
    
    # Define properties
    width = 30
    length = 30
    height = 20
    radius1 = 20
    radius2 = -20
    
    # Generate left-arc
    an1 = asin(width/(2*radius1))
    ac1 = linspace(-an1, an1, 64)
    xc1 = linspace(-length/2, -length/2, 64)
    yc1 = radius1*sin(ac1)
    zc1 = radius1*(1-cos(ac1))
    zc1 += height

    # Generate right-arc
    an2 = asin(width/(2*radius2))
    ac2 = linspace(-an2, an2, 64)
    xc2 = linspace(length/2, length/2, 64)
    yc2 = radius2*sin(ac2)
    zc2 = radius2*(1-cos(ac2))
    zc2 += height
    
    # Generate grid
    xGrid = zeros((64,64))
    yGrid = zeros((64,64))
    zGrid = zeros((64,64))
    for i in range(64):
        xGrid[i,:] = linspace(xc1[i],xc2[i],64);
        yGrid[i,:] = linspace(yc1[i],yc2[i],64);
        zGrid[i,:] = linspace(zc1[i],zc2[i],64);

    # Convert grid to spline geometry
    array2 = TColgp_Array2OfPnt(1, 64, 1, 64)
    for i in range(64):
        for j in range(64):
            p = gp_Pnt()
            p.SetCoord(xGrid[i, j], yGrid[i, j], zGrid[i, j])
            array2.SetValue(i + 1, j + 1, p)
    bspline = GeomAPI_PointsToBSplineSurface(array2, 3, 8, GeomAbs_C2, 1e-3)  # 1e-3 is the accuracy in mm
    
    # Convert to face 
    face = BRepBuilderAPI_MakeFace(bspline.Surface(), 1e-6).Shape()  # 1e-6 is the accuracy in mm

    # Extrude face upward
    vector = gp_Vec(0, 0, 100)
    prism = BRepPrimAPI_MakePrism(face, vector).Shape()
    
    # Create base box
    thick =  amax([amax(zc1), amax(zc2)])
    box = BRepPrimAPI_MakeBox(width, length, height+thick).Shape()

    # Apply Transform to Box
    transform = gp_Trsf()
    transform.SetTranslationPart(gp_Vec(-width/2, -length/2, 0))
    location = TopLoc_Location(transform)
    box.Location(location)

    # Finally, cut box with prism
    solid = TopoDS_Shape(BRepAlgoAPI_Cut(box, prism).Shape())
    
    # Display result
    occ.canva._display.Draw(solid, True)  

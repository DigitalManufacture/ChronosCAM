__author__ = "Anthony Beaucamp"
__copyright__ = "Copyright (C) 2022 Kyoto University"
__license__ = "Public Domain"
__version__ = "1.0"

from numpy import *
from scipy.spatial import Delaunay

from OCC.Core.gp import *
from OCC.Core.BRepAdaptor import *
from OCC.Core.BRepLProp import *
from OCC.Tools.Dialogs import *
from OCC.Tools.Plotly import *

def ComputePrincipalCurvatures(occ):
    
    # Check that sections are available
    if occ.sections == []:
        warndlg(occ, "Please generate sections first!", "Warning")
        return

    # Initialize some variables
    fi = -1             # Index of current face
    minDir = gp_Dir()   # Direction of min. curvature
    maxDir = gp_Dir()   # Direction of max. curvature

    # Initialize plotting variables?
    if occ.plotting:
        x = []; y = []; z = []
        xMin = []; yMin = []; zMin = []
        xMax = []; yMax = []; zMax = []
        cMin = []; cMax = []

    # Process sections
    occ.curvature = []
    for section in occ.sections:
        
        # Unpack section data
        points = section[0]
        normals = section[1]
        tangents = section[2]
        fuv = section[3]

        # Process points
        minCurvVal = zeros(points[0].shape)
        maxCurvVal = zeros(points[0].shape)
        minCurvDir = zeros(points.shape)
        maxCurvDir = zeros(points.shape)
        for i in range(len(points[0])):
            
            # Update face?
            if fuv[0,i] != fi:
                fi = fuv[0,i]
                face = occ.selection[int(fi)]
                props = BRepLProp_SLProps(BRepAdaptor_Surface(face), 1, 0.000001)
            
            # Set coordinates of UV point
            props.SetParameters(fuv[1,i], fuv[2,i])
            
            # Get curvature values
            minCurvVal[i] = props.MinCurvature()
            maxCurvVal[i] = props.MaxCurvature()
            
            # Get curvature directions
            props.CurvatureDirections(maxDir, minDir)
            minCurvDir[0,i] = minDir.X()
            minCurvDir[1,i] = minDir.Y()
            minCurvDir[2,i] = minDir.Z()
            maxCurvDir[0,i] = maxDir.X()
            maxCurvDir[1,i] = maxDir.Y()
            maxCurvDir[2,i] = maxDir.Z()
            
        # Store data
        occ.curvature.append([points, minCurvVal, maxCurvVal, minCurvDir, maxCurvDir])
        
        # Pack data for plotting?
        if occ.plotting:
            x = append(x, points[0])
            y = append(y, points[1])
            z = append(z, points[2])
            xMin = append(xMin, minCurvDir[0])
            yMin = append(yMin, minCurvDir[1]) 
            zMin = append(zMin, minCurvDir[2]) 
            cMin = append(cMin, minCurvVal)
            xMax = append(xMax, maxCurvDir[0])
            yMax = append(yMax, maxCurvDir[1]) 
            zMax = append(zMax, maxCurvDir[2]) 
            cMax = append(cMax, maxCurvVal)
    
    # Plot results?
    if occ.plotting:
        vertices = c_[x.reshape(-1,1), y.reshape(-1,1)]
        tri = Delaunay(vertices)
        faces = tri.simplices
        
        vertices = c_[vertices, z.reshape(-1,1)]
        colorMin = (cMin[faces[:,0]] + cMin[faces[:,1]] + cMin[faces[:,2]])/3
        colorMax = (cMax[faces[:,0]] + cMax[faces[:,1]] + cMax[faces[:,2]])/3
        
        figure(title='Min. Curvature', visible=False)
        mesh(vertices, faces, colorMin, colormap='jet')
        colorbar('Curvature')
        quiver3(x, y, z+1, xMin, yMin, zMin, color='g')                    
        labels('X (mm)', 'Y (mm)', 'Z (mm)'); grid('on')
        view(eye=[0, 0, 2.5], dragmode=False, orthographic=True)
        showfig()            

        figure(title='Max. Curvature', visible=False)
        mesh(vertices, faces, colorMax, colormap='jet')
        colorbar('Curvature')
        quiver3(x, y, z+1, xMax, yMax, zMax, color='g')                    
        labels('X (mm)', 'Y (mm)', 'Z (mm)'); grid('on')
        view(eye=[0, 0, 2.5], dragmode=False, orthographic=True)
        showfig()            

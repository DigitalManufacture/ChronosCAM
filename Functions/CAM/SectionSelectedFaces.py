__author__ = "Anthony Beaucamp"
__copyright__ = "Copyright (C) 2022 Kyoto University"
__license__ = "Public Domain"
__version__ = "1.0"

from OCC.Core.gp import *
from OCC.Core.BRepAlgoAPI import *
from OCC.Tools.Dialogs import *
from OCC.Tools.Topology import *
from OCC.Tools.Plotly import *

def SectionSelectedFaces(occ):

    sampling = 0.5
    
    # Get selected faces
    occ.selection = occ.GetSelection()     
    if occ.selection == []:
        warndlg(occ, "Please select faces first!", "Warning")
        return
    shell = face2shell(occ.selection)

    # Create figure?        
    if occ.plotting:
        figure(title='Sections', visible=False)
                
    # Section surface
    occ.sections = []
    for y in linspace(-14,14,15):
        point = gp_Pnt(0, y, 0)
        direction = gp_Dir(0, 1, 0)
        plane = gp_Pln(point, direction)

        section = BRepAlgoAPI_Section(shell, plane)
        points, normals, tangents, fuv = section2points(section, shell, direction, sampling)
        occ.sections.append([points, normals, tangents, fuv])
        
        if occ.plotting:
            plot3(points[0], points[1], points[2], color='r')            
        occ.canva._display.Draw(section.Shape(), True)        

    # Add axis labels?        
    if occ.plotting:
        labels('X (mm)', 'Y (mm)', 'Z (mm)')
        grid('on')
        showfig()

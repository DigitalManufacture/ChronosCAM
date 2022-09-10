__author__ = "Anthony Beaucamp"
__copyright__ = "Copyright (C) 2020 Kyoto University"
__license__ = "Public Domain"
__version__ = "1.0"

import sys, os, os.path

from math import *
from numpy import *
from numpy.linalg import *
import ctypes

from OCC.Core.gp import *
from OCC.Core.BRep import *
from OCC.Core.BRepAdaptor import *
from OCC.Core.BRepLProp import *
from OCC.Core.BRepTools import *
from OCC.Core.BRepExtrema import *
from OCC.Core.TopAbs import *
from OCC.Core.TopLoc import *
from OCC.Utils.Topology import *
from OCC.Core.TopExp import *
from OCC.Core.GeomLib import *
from OCC.Core.Geom import *

from PyQt5 import QtGui, QtCore, QtWidgets

QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)  # enable highdpi scaling
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)  # use highdpi icons

try:
    from PyQt5 import QtWebEngineWidgets
except:
    try:
        from PyQt5.QtWebEngineWidgets import *
    except:
        app = QtWidgets.QApplication.instance()
        if app is not None:
            import sip

            app.quit()
            sip.delete(app)
        from PyQt5 import QtWebEngineWidgets

        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
        app = QtWidgets.qApp = QtWidgets.QApplication(sys.argv)

from PyQt5.QtWidgets import QMainWindow, QApplication
from OCC.Utils.Topology import *
from OCC.Core.Prs3d import Prs3d_TextAspect, Prs3d_Text
from OCC.Display.backend import load_backend, get_qt_modules

load_backend("qt-pyqt5")
from OCC.Display.OCCViewer import Viewer3d, to_string

from OCC.Display.qtDisplay import qtViewer3d
from OCC.Core.Graphic3d import *
from OCC.Core.Quantity import *
from OCC.Core.Aspect import *
from OCC.Core.MeshVS import *
from OCC.Core.AIS import *
from OCC.Core import VERSION
import OCC.Core.CTM as ctm
from win32gui import SetWindowPos
import win32con

from OCC.Tools.Dictionary import *


class point(object):
    def __init__(self, obj=None):
        self.x = 0
        self.y = 0
        if obj is not None:
            self.set(obj)

    def set(self, obj):
        self.x = obj.x()
        self.y = obj.y()


def occhold():
    app = QtWidgets.QApplication.instance()
    app.exec_()


def occviewer(showGrid=True, size=None):
    """!
    Creates the graphical interface for the OCC viewer.
    @param showGrid: boolean flag to specify if the grid should be rendered.
    @return: tupple containing the windows handle and a handle to the start_display function
    """
    global display, win, app

    app = QtWidgets.QApplication.instance()
    if not app:  # create QApplication if it doesnt exist
        app = QtWidgets.QApplication(sys.argv)

    win = MainWindow(size)
    SetWindowPos(win.winId(), win32con.HWND_TOPMOST,  # = always on top. only reliable way to bring it to the front on windows
                 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW)
    SetWindowPos(win.winId(), win32con.HWND_NOTOPMOST,  # disable the always on top, but leave window at its top position
                 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW)
    win.raise_()
    win.show()
    win.activateWindow()
    win.canva.InitDriver()
    win.canva.qApp = app
    display = win.canva._display
    if showGrid:
        Grid(display)
    display.SetBackgroundImage(background())

    return display


class Viewer(Viewer3d):
    """!
    Extends the Viewer3d class
    """

    black = 0
    blue = 1
    grey = 2
    silver = 3
    cream = 4
    brown = 20
    purple = 26
    blood = 27
    red = 30
    cyan = 38
    green = 45
    orange = 48
    pink = 102
    white = 119
    yellow = 123

    def __init__(self, canva):
        """!
        Default constructor
        @param window_handle: handle to window
        """
        Viewer3d.__init__(self)
        self.selection = []
        self.lable_list = []
        self.shapeLibrary = Dictionary()
        self.selectMode = 'face'
        self.canva = canva

    def Erase(self):
        """!
        Removes all objects from the viewer.
        """
        self.selection = []
        self.shapeLibrary = Dictionary()
        self.Context.EraseAll(True)

    def Label(self, point, label, height=None, col=None, update=True):
        """!
        Places a text object on the viewer
        :param point: a gp_Pnt or gp_Pnt2d instance
        :param label:  a string
        :param height:
        :param col: triple with the range 0-1
        :param update:
        :return:
        """

        aPresentation = Graphic3d_Structure(self._struc_mgr)
        text_aspect = Prs3d_TextAspect()

        if col is not None:
            text_aspect.SetColor(Quantity_Color(col))
        if height is not None:
            text_aspect.SetHeight(height)
        if isinstance(point, gp_Pnt2d):
            point = gp_Pnt(point.X(), point.Y(), 0)
        Prs3d_Text.Draw(aPresentation, text_aspect, to_string(label), point)
        aPresentation.Display()

        # @TODO: it would be more coherent if a AIS_InteractiveObject is returned
        if update:
            self.Repaint()
        return aPresentation

    def Draw(self, shape, selectable=True, color=None, texture={}):
        """!
        Renders a shape with its associated colour and texture properties.
        @param shape: mesh or shape object
        @param selectable: boolean
        @param color: Quantity_Color
        @param texture: list[texture] , list of textures associated with this shape
        """
        # if type(shape) is ctm.Mesh:
        #     aDS = ctm.CTM_MeshVSLink(shape)
        #     aMeshVS = MeshVS_Mesh(True)
        #     DMF = 1 # to wrap!
        #     MeshVS_BP_Mesh       =  5 # To wrap!
        #     aPrsBuilder = MeshVS_MeshPrsBuilder(aMeshVS.GetHandle() ) # ,DMF, aDS.GetHandle(),0,MeshVS_BP_Mesh)
        #     # aPrsBuilder = MeshVS_MeshPrsBuilder(aMeshVS.GetHandle() ) # ,DMF, aDS.GetHandle(),0,MeshVS_BP_Mesh)
        #     # set the mesh renderer
        #     aMeshVS.SetDataSource(aDS.GetHandle())
        #     aMeshVS.AddBuilder(aPrsBuilder.GetHandle(),True)
        #     # # # display the mesh edges in black
        #     mesh_drawer = aMeshVS.GetDrawer().GetObject()
        #     mesh_drawer.SetBoolean(MeshVS_DA_DisplayNodes, False)
        #     mesh_drawer.SetBoolean(MeshVS_DA_ShowEdges, False)
        #     mesh_drawer.SetBoolean(MeshVS_DA_SmoothShading, True)
        #     mesh_drawer.SetBoolean(MeshVS_DA_Reflection, True)
        #     MeshVS_DMF_HilightPrs = int(0x0400)
        #     MeshVS_DMF_Wireframe = int(0x0001)
        #     MeshVS_DMF_Shading = int(0x0002)
        #     MeshVS_SMF_Face = int(0x0008)
        #     # if wireframe:
        #         # aMeshVS.SetDisplayMode(MeshVS_DMF_Wireframe) # 1 -> wireframe, 2 -> shaded
        #     # else:
        #     aMeshVS.SetDisplayMode(MeshVS_DMF_Shading) # 1 -> wireframe, 2 -> shaded
        #     aMeshVS.SetSelectionMode(MeshVS_SMF_Face)
        #     drawer = aMeshVS.GetHandle().GetObject().GetDrawer().GetObject()
        #
        #     #drawer.SetMaterial(MeshVS_DA_FrontMaterial, Graphic3d_MaterialAspect(22))  # Mesh Color BLUE
        #     #drawer.SetMaterial(MeshVS_DA_FrontMaterial, Graphic3d_MaterialAspect(0))  # Mesh Color BRIGHT YELLOW
        #     # drawer.SetMaterial(MeshVS_DA_FrontMaterial, Graphic3d_MaterialAspect(1))  # Mesh Color DARK YELLOW
        #     #drawer.SetMaterial(MeshVS_DA_FrontMaterial, Graphic3d_MaterialAspect(12))  # Mesh Color BLUE
        #
        #     interact = aMeshVS.GetHandle()
        #     if shape in self.selection:
        #         drawer.SetMaterial(MeshVS_DA_FrontMaterial, Graphic3d_MaterialAspect(Graphic3d_NOM_ALUMINIUM))
        #         self.Context.SetMaterial(interact, Graphic3d_NOM_ALUMINIUM)
        #     else:
        #         drawer.SetMaterial(MeshVS_DA_FrontMaterial, Graphic3d_MaterialAspect(1))
        #         self.Context.SetMaterial(interact, Graphic3d_NOM_PLASTIC)
        #     self.Context.Display(aMeshVS.GetHandle())
        #     interact = aMeshVS.GetHandle()
        #     shapeInteracts = Dictionary()
        #     shapeInteracts[shape] = interact
        #     self.shapeLibrary[shape] = shapeInteracts
        #     return interact
        # else:
            
        # Add to tree
        #self.canva.window.AddTree(shape)                  
            
        index = 0
        faces = Topo(shape).faces()
        if selectable and faces.__length_hint__() > 0:
            # Display each Face independently
            shapeInteracts = Dictionary()
            for face in faces:
                # if texture and len(texture) > index and texture[index]:
                if texture and face in texture.keys() and texture[face]:
                    interact = self.DisplayShape(face, texture=texture[face][3], color=Quantity_NOC_WHITE)[0]
                else:
                    interact = self.DisplayShape(face)[0]
                shapeInteracts[face] = interact
                if face in self.selection:
                    self.Context.SetMaterial(interact, Graphic3d_MaterialAspect(Graphic3d_NOM_ALUMINIUM), True)
                else:
                    self.Context.SetMaterial(interact, Graphic3d_MaterialAspect(Graphic3d_NOM_PLASTIC), True)
                if color is not None:
                    self.Context.SetColor(interact, color)
                index += 1
            self.shapeLibrary[shape] = shapeInteracts
            self.Repaint()
            self.FitAll()
            return shapeInteracts
        else:            
            # Display whole object at-once
            if texture and type(texture) not in [list, dict]:
                interact = self.DisplayShape(shape, texture=texture)
            else:
                interact = self.DisplayShape(shape, True)
                
            # Set Material/Color
            if shape in self.selection:
                if type(interact) is list:
                    for i in interact:
                        self.Context.SetMaterial(i, Graphic3d_MaterialAspect(Graphic3d_NOM_ALUMINIUM), True)
                else:
                    self.Context.SetMaterial(interact, Graphic3d_MaterialAspect(Graphic3d_NOM_ALUMINIUM), True)
            else:
                if type(interact) is list:
                    for i in interact:
                        self.Context.SetMaterial(i, Graphic3d_MaterialAspect(Graphic3d_NOM_PLASTIC), True)
                else:
                    self.Context.SetMaterial(interact, Graphic3d_MaterialAspect(Graphic3d_NOM_PLASTIC), True)
            if color is not None:
                if type(interact) is list:
                    for i in interact:
                        self.Context.SetColor(i, Quantity_Color(color), True)
                else:
                    self.Context.SetColor(interact, Quantity_Color(color), True)
            return interact

    def Update(self, interactive, transform, color=None, redraw=True):
        """!
        Updates the interactive object's transformation and color properties
        @param interactive:  object to manipulate
        @param transform: transform object (rotation, translation)
        @param color: Quantity_Color object
        @param redraw: boolean
        """
        location = TopLoc_Location(transform)
        if type(interactive) is Dictionary:
            for face in interactive:
                self.Context.SetLocation(interactive[face], location)
                if color is not None:
                    self.Context.SetColor(interactive[face], Quantity_Color(color), True)
        elif type(interactive) is list:
            for face in interactive:
                self.Context.SetLocation(face, location)
                if color is not None:
                    self.Context.SetColor(face, Quantity_Color(color), True)
        else:
            self.Context.SetLocation(interactive, location)
            if color is not None:
                self.Context.SetColor(interactive, Quantity_Color(color), True)
        if redraw:
            self.Context.UpdateCurrentViewer()
            self.FitAll()

    def Refresh(self):
        """!
        Refreshes the viewer
        """
        self.Context.UpdateCurrentViewer()

    def Redraw(self, objects):
        """!
        Clears the screen,redraws and fits all objects to the viewer
        @param objects: list of objects to redraw
        """
        self.Erase()
        for object in objects:
            self.Draw(object)
        self.FitAll()

    def GetSelectedMeshes(self):
        """!
        Get the selected meshes
        @returns: returns a list of selected meshes
        """
        meshList = []
        for interact in self.selection:
            if type(interact) is ctm.Mesh:
                meshList.append(interact)
        return meshList

    def GetSelection(self, shape):
        """!
        Get a list of all selected faces from shape object
        @param shape: shape object to inspect
        @return: return list of all selected faces
        """
        selectedFaces = []
        if shape is not None:
            shapeInteracts = self.shapeLibrary[shape]
            for face in shapeInteracts:
                if face in self.selection:
                    selectedFaces.append(face)
        else:
            for shape in self.shapeLibrary:
                if shape.__class__.__name__ == "TopoDS_Shape":
                    for face in self.shapeLibrary[shape]:
                        if face in self.selection:
                            selectedFaces.append(face)            
                else:
                    face = self.shapeLibrary[shape]
                    if face in self.selection:
                        selectedFaces.append(face)                                
        return selectedFaces

    def AddSelection(self, interact, meshInteract=None, face=None, index=None, object=None):
        """!
        Adds an interact shape or mesh object to the list of selected objects
        @param interact: interactive shape object
        @param meshInteract: interactive mesh object
        @param face:
        @param index:
        @param object:
        """
        # Add to Selection
        if meshInteract is not None:
            drawer = meshInteract.GetDrawer()
            drawer.SetMaterial(MeshVS_DA_FrontMaterial, Graphic3d_MaterialAspect(Graphic3d_NOM_ALUMINIUM), True)
            self.Context.SetMaterial(interact, Graphic3d_MaterialAspect(Graphic3d_NOM_ALUMINIUM), True)
        else:
            self.Context.SetMaterial(interact, Graphic3d_MaterialAspect(Graphic3d_NOM_ALUMINIUM), True)
        self.Context.UpdateCurrentViewer()

    def RemSelection(self, interact, meshInteract=None, face=None, index=None, object=None):
        """!
        Removes an interact shape or mesh interact from the list of selected objects
        @param interact: interactive shape object
        @param meshInteract: interactive mesh object
        @param face:
        @param index:
        @param object:
        """
        # Rem from Selection
        if meshInteract is not None:
            drawer = meshInteract.GetDrawer()
            drawer.SetMaterial(MeshVS_DA_FrontMaterial, Graphic3d_MaterialAspect(1), True)
            self.Context.SetMaterial(interact, Graphic3d_MaterialAspect(Graphic3d_NOM_PLASTIC), True)
        else:
            self.Context.SetMaterial(interact, Graphic3d_MaterialAspect(Graphic3d_NOM_PLASTIC), True)
        self.Context.UpdateCurrentViewer()

    def FaceIndices(self):
        """!
        Get face from interactive
        """
        # Get Face from Interactive
        ind = 0
        pnt = gp_Pnt()
        if self.lable_list:
            for lable in self.lable_list:
                lable.Erase()
            self.lable_list = []
            self.Refresh()
        else:
            for aShape in self.shapeLibrary.keys():
                if shape is not None:
                    for aFace in self.shapeLibrary[aShape].keys():
                        # Compute centre of face
                        surface = BRepAdaptor_Surface(aFace)
                        uMin, uMax, vMin, vMax = breptools_UVBounds(aFace)
                        surface.D0((uMin + uMax) / 2, (vMin + vMax) / 2, pnt)
                        # Compute normal at centre
                        surf = BRepLProp_SLProps(surface, 1, 0.000001)
                        surf.SetParameters((uMin + uMax) / 2, (vMin + vMax) / 2)
                        evalNor = surf.Normal()
                        if aFace.Orientation() == TopAbs_REVERSED:
                            evalNor.Reverse()
                        # Draw face number
                        pnt.SetX(pnt.X() + evalNor.X())
                        pnt.SetY(pnt.Y() + evalNor.Y())
                        pnt.SetZ(pnt.Z() + evalNor.Z())
                        self.lable_list.append(self.Label(pnt, str(ind), height=30, col=Quantity_NOC_GREEN))
                        ind += 1

    def SelectFaces(self, indices):
        """!
        Select all the faces
        @param indices: list of face indices
        """
        for index in indices:
            self.SelectFace(index)

    def SelectFace(self, index):
        """!
        Selects a face by its index.
        @param index: int
        """
        ind = 0
        for aShape in self.shapeLibrary.keys():
            if shape is not None:
                for aFace in self.shapeLibrary[aShape].keys():
                    if ind == index:
                        if aFace not in self.selection:
                            shapeInteracts = self.shapeLibrary[aShape]
                            interact = shapeInteracts[aFace]
                            self.AddSelection(interact, face=aFace)
                            self.selection.append(aFace)
                    ind += 1

    def GetFace(self, index):
        ind = 0
        for aShape in self.shapeLibrary.keys():
            if ind == index:
                return aShape
            ind += 1    
        return None
    
    def DeSelect_All(self):
        """!Deselect all object in the viewer"""
        self.selection = []
        self.Context.EraseAll(True)
        for shape in self.shapeLibrary.keys():
            self.Draw(shape)

    def ToggleFace(self, interact, shape=None):
        """!
        Toggles On or Off the selected faces for an interact object
        @param interact:
        @param shape:
        """

        # Get Face from Interactive
        for aShape in self.shapeLibrary.keys():
            if shape is not None:
                # try:
                # shape = topods_Face(shape)
                for aFace in self.shapeLibrary[aShape].keys():
                    if aFace == shape:
                        if aFace in self.selection:
                            self.RemSelection(interact, face=aFace)
                            self.selection.remove(aFace)
                            return
                        else:
                            self.AddSelection(interact, face=aFace)
                            self.selection.append(aFace)
                            return
                # except:
                #     pass

            for i, aMesh in enumerate(self.shapeLibrary[aShape].keys()):
                aninteract = self.shapeLibrary[aShape][aMesh]
                if aninteract == interact:
                    if aMesh in self.selection:
                        self.RemSelection(interact, meshInteract=aninteract)
                        self.selection.remove(aMesh)
                        return
                    else:
                        self.AddSelection(interact, meshInteract=aninteract)
                        self.selection.append(aMesh)
                        return

    def Toggle(self):
        """!Change face/mesh display material"""
        # Change face/mesh display material
        self.Context.Select(True)
        self.Context.InitSelected()
        while self.Context.MoreSelected():
            self.ToggleFace(self.Context.SelectedInteractive(), shape=self.Context.SelectedShape())
            self.Context.NextSelected()


class Canva(qtViewer3d):
    """!
    Extends qtViewer3d
    """
    Select = True
    DynaZoom = False
    DynaRotate = False
    DynaPan = False

    def InitDriver(self):
        """!driver initialization"""
        self._display = Viewer(self)
        self._display.Create(window_handle=int(self.winId()), parent=self)
        self._display.display_triedron()
        self._display.SetModeShaded()
        self._inited = True
        # drawer = self._display.GetContext().DefaultDrawer()
        # drawer.SetColor(Quantity_Color(Quantity_NOC_WHITE))
        # self._display.GetContext().SetSelectionStyle(drawer)
        # dict mapping keys to functions
        # self._SetupKeyMap()

    def mouseReleaseEvent(self, event):
        """!
        Mouse release event handler
        @param event: event
        """
        pt = point(event.pos())
        if event.button() == QtCore.Qt.LeftButton:
            pt = point(event.pos())
            # Face selection
            if self.Select and self._display.selectMode == 'face':
                self._display.Select(pt.x, pt.y)
                self._display.Toggle()
            # Edge selection
            if self.Select and self._display.selectMode == 'edge':
                self._display.ShiftSelect(pt.x, pt.y)

        elif event.button() == QtCore.Qt.RightButton:
            if self._zoom_area:
                [Xmin, Ymin, dx, dy] = self._drawbox
                self._display.ZoomArea(Xmin, Ymin, Xmin + dx, Ymin + dy)
                self._zoom_area = False

    def mouseMoveEvent(self, evt):
        """!
        Mouse move event handler
        @param evt: event
        """
        pt = point(evt.pos())
        try:
            scale = float(ctypes.windll.shcore.GetScaleFactorForDevice(0)) / 100
        except:
            scale = 0.666        
        pt.x = int(pt.x*scale)
        pt.y = int(pt.y*scale)
        self.window.statusBar().showMessage("X:" + str(pt.x) + " Y:" + str(pt.y))
        buttons = int(evt.buttons())
        modifiers = evt.modifiers()
        # ROTATE
        if (buttons == QtCore.Qt.RightButton or (buttons == QtCore.Qt.LeftButton and self.DynaRotate)):
            dx = pt.x - self.dragStartPosX
            dy = pt.y - self.dragStartPosY
            self.cursor = "rotate"
            self._display.Rotation(pt.x, pt.y)
            self._drawbox = False
        # DYNAMIC ZOOM
        elif (buttons == QtCore.Qt.LeftButton and self.DynaZoom):
            self._display.Repaint()
            self.cursor = "zoom"
            self._display.DynamicZoom(abs(self.dragStartPosX), abs(self.dragStartPosY), abs(pt.x), abs(pt.y))
            self.dragStartPosX = pt.x
            self.dragStartPosY = pt.y
            self._drawbox = False
        # PAN
        elif (buttons == QtCore.Qt.LeftButton and self.DynaPan):
            dx = pt.x - self.dragStartPosX
            dy = pt.y - self.dragStartPosY
            self.cursor = "pan"
            self.dragStartPosX = pt.x
            self.dragStartPosY = pt.y
            self._display.Pan(dx, -dy)
            self._drawbox = False
        # DRAW BOX
        # elif (buttons == QtCore.Qt.RightButton and modifiers == QtCore.Qt.ShiftModifier):  # ZOOM WINDOW
        #     self._zoom_area = True
        #     self.cursor = "zoom-area"
        #     self.DrawBox(evt)
        # elif (buttons == QtCore.Qt.LeftButton and modifiers == QtCore.Qt.ShiftModifier):  # SELECT AREA
        #     self._select_area = True
        #     self.DrawBox(evt)
        else:
            self.cursor = "arrow"
            self._drawbox = False
            self._display.MoveTo(pt.x, pt.y)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, size, *args):
        """
        Default constructor. Create OpenGL render window
        :param args:
        """
        # Create OpenGL render window
        QtWidgets.QMainWindow.__init__(self, *args)
        
        self.canva = Canva(self)
        self.canva.window = self
        # self.canva.setGeometry(QtCore.QRect(200, 90, 1080, 678))
        self.setCentralWidget(self.canva)
        
        # ##################################################                
        # self.tree = QtWidgets.QTreeWidget(self)
        # self.tree.setHeaderHidden(True)

        # font =  self.tree.font()
        # font.setPointSize(8);
        # self.tree.setFont(font)                

        # ##################################################        
        # self.inspector = QtWidgets.QTableWidget(self)
        # self.inspector.setRowCount(3)
        # self.inspector.setColumnCount(2)
        # self.inspector.verticalHeader().setVisible(False)
        
        # font =  self.inspector.font()
        # font.setPointSize(8);
        # self.inspector.setFont(font)                

        # font =  self.inspector.horizontalHeader().font()
        # font.setPointSize(8);
        # self.inspector.horizontalHeader().setFont(font)                

        # self.inspector.setHorizontalHeaderLabels(["Property", "Value"])
        # self.inspector.setItem(0, 0, QtWidgets.QTableWidgetItem("Position"))
        # self.inspector.setItem(1, 0, QtWidgets.QTableWidgetItem("Rotation"))
        # self.inspector.setItem(2, 0, QtWidgets.QTableWidgetItem("Scale"))
        # self.inspector.resizeColumnsToContents()        
                
        # ##################################################                
        # central_widget = QtWidgets.QWidget()
        # self.setCentralWidget(central_widget)

        # column1 = QtWidgets.QWidget()
        # layout1 = QtWidgets.QGridLayout(column1)
        # layout1.setSpacing(0)
        # layout1.setContentsMargins(0,0,0,0)
        # layout1.addWidget(self.tree, 0, 0)
        # layout1.addWidget(self.inspector, 1, 0)
        
        # layout2 = QtWidgets.QGridLayout(central_widget)
        # layout2.setSpacing(0)
        # layout2.setContentsMargins(0,0,0,0)
        # layout2.addWidget(column1, 0, 0)
        # layout2.addWidget(self.canva, 0, 1)
        # layout2.setColumnStretch(0, 1)
        # layout2.setColumnStretch(1, 5)

        # #######################################
        
        self.setWindowTitle("pythonOCC-%s 3d viewer ('qt' backend)" % VERSION)

        try:
            scale = float(ctypes.windll.shcore.GetScaleFactorForDevice(0)) / 100
        except:
            scale = 0.666        
        if size is None: 
            self.resize(int(1400/scale), int(900/scale))
        else:
            self.resize(int(size[0]/scale), int(size[1]/scale))
            
        if not sys.platform == 'darwin':
            self.menu_bar = self.menuBar()
        else:
            # create a parentless menubar
            # see: http://stackoverflow.com/questions/11375176/qmenubar-and-qmenu-doesnt-show-in-mac-os-x?lq=1
            # noticeable is that the menu ( alas ) is created in the
            # topleft of the screen, just next to the apple icon
            # still does ugly things like showing the "Python" menu in bold
            self.menu_bar = QtWidgets.QMenuBar()
            
        self._menus = {}
        self._menu_methods = {}
        
        # place the window in the center of the screen, at half the screen size
        self.centerOnScreen()
        self.statusBar().showMessage("")

        # Create Toolbar
        select = QtWidgets.QAction(QtGui.QIcon(icon('select.png')), 'Select', self)
        select.triggered.connect(self._select)

        pan = QtWidgets.QAction(QtGui.QIcon(icon('pan.png')), 'Pan', self)
        pan.triggered.connect(self._pan)

        rotate = QtWidgets.QAction(QtGui.QIcon(icon('rotate.png')), 'Rotate', self)
        rotate.triggered.connect(self._rotate)

        zoom = QtWidgets.QAction(QtGui.QIcon(icon('zoom.png')), 'Zoom', self)
        zoom.triggered.connect(self._zoom)

        indices = QtWidgets.QAction(QtGui.QIcon(icon('indices.png')), 'Show Face Indices', self)
        indices.triggered.connect(self._indices)

        wireframe = QtWidgets.QAction(QtGui.QIcon(icon('wireframe.png')), 'Wireframe', self)
        wireframe.triggered.connect(self.Wireframe)

        shaded = QtWidgets.QAction(QtGui.QIcon(icon('shaded.png')), 'Shaded', self)
        shaded.triggered.connect(self.Shaded)

        topview = QtWidgets.QAction(QtGui.QIcon(icon('topview.png')), 'Top View', self)
        topview.triggered.connect(self.View_Top)

        bottomview = QtWidgets.QAction(QtGui.QIcon(icon('bottomview.png')), 'Bottom View', self)
        bottomview.triggered.connect(self.View_Bottom)

        leftview = QtWidgets.QAction(QtGui.QIcon(icon('leftview.png')), 'Left View', self)
        leftview.triggered.connect(self.View_Left)

        rightview = QtWidgets.QAction(QtGui.QIcon(icon('rightview.png')), 'Right View', self)
        rightview.triggered.connect(self.View_Right)

        frontview = QtWidgets.QAction(QtGui.QIcon(icon('frontview.png')), 'Front View', self)
        frontview.triggered.connect(self.View_Front)

        rearview = QtWidgets.QAction(QtGui.QIcon(icon('rearview.png')), 'Rear View', self)
        rearview.triggered.connect(self.View_Rear)

        resetview = QtWidgets.QAction(QtGui.QIcon(icon('resetview.png')), 'Reset View', self)
        resetview.triggered.connect(self.View_Reset)

        selectall = QtWidgets.QAction(QtGui.QIcon(icon('selectall.png')), 'Select All', self)
        selectall.triggered.connect(self.Select_All)

        deselectall = QtWidgets.QAction(QtGui.QIcon(icon('deselectall.png')), 'Deselect All', self)
        deselectall.triggered.connect(self.DeSelect_All)

        self.toolbar = self.addToolBar('Toolbar')
        self.toolbar.addAction(select)
        self.toolbar.addAction(pan)
        self.toolbar.addAction(rotate)
        self.toolbar.addAction(zoom)
        self.toolbar.addSeparator()
        self.toolbar.addAction(indices)
        self.toolbar.addAction(selectall)
        self.toolbar.addAction(deselectall)
        self.toolbar.addSeparator()
        self.toolbar.addAction(wireframe)
        self.toolbar.addAction(shaded)
        self.toolbar.addSeparator()
        self.toolbar.addAction(topview)
        self.toolbar.addAction(bottomview)
        self.toolbar.addAction(leftview)
        self.toolbar.addAction(rightview)
        self.toolbar.addAction(frontview)
        self.toolbar.addAction(rearview)
        self.toolbar.addAction(resetview)
        self.toolbar.addSeparator()
        
    def AddTree(self, shape):
        self.tree.clear()
        data = {"Project A": ["file_a.py", "file_a.txt", "something.xls"],
                "Project B": ["file_b.csv", "photo.jpg"],
                "Project C": []}    
        items = []
        for key, values in data.items():
            item = QtWidgets.QTreeWidgetItem([key])
            #for value in values:
            #    ext = value.split(".")[-1].upper()
            #    child = QtWidgets.QTreeWidgetItem([value, ext])
            #    item.addChild(child)
            items.append(item)
        self.tree.insertTopLevelItems(0, items)        
        

    def centerOnScreen(self):
        resolution = QtWidgets.QDesktopWidget().screenGeometry()
        self.move(int((resolution.width() / 2) - (self.frameSize().width() / 2)), int((resolution.height() / 2) - (self.frameSize().height() / 2)))

    def add_menu(self, menu_name):
        _menu = self.menu_bar.addMenu("&" + menu_name)
        self._menus[menu_name] = _menu

    def add_menu_item(self, menu_name, function_name, _callable):
        assert callable(_callable), 'the function supplied is not callable'
        _action = QtWidgets.QAction(function_name, self)
        _action.triggered.connect(_callable)
        self._menus[menu_name].addAction(_action)

    def add_tool_bar(self, iconName, name, function, toggable=False):
        icon = QtGui.QIcon(iconName)
        toolBar = QtWidgets.QAction(icon, name.title(), self)
        toolBar.triggered.connect(function)
        if toggable:
            toolBar.setCheckable(True)
        self.toolbar.addAction(toolBar)

    def _select(self, event):
        self.SetTogglesToFalse()
        self.canva.Select = True
        self._refreshui()

    def _zoom(self, event):
        self.SetTogglesToFalse()
        self.canva.DynaZoom = True
        self._refreshui()

    def _pan(self, event):
        self.SetTogglesToFalse()
        self.canva.DynaPan = True
        self._refreshui()

    def _rotate(self, event):
        self.SetTogglesToFalse()
        self.canva.DynaRotate = True
        self._refreshui()

    def _indices(self, event):
        self.canva._display.FaceIndices()

    def View_Top(self):
        """!Toggle top view"""
        self.canva._display.View_Top()

    def View_Bottom(self):
        """!Toggle bottom view"""
        self.canva._display.View_Bottom()

    def View_Left(self):
        """!Toggle left view"""
        self.canva._display.View_Left()

    def View_Right(self):
        """!Toggle right view"""
        self.canva._display.View_Right()

    def View_Front(self):
        """!Toggle front view"""
        self.canva._display.View_Front()

    def View_Rear(self):
        """!Toggle rear view"""
        self.canva._display.View_Rear()

    def View_Reset(self):
        """!Reset to the default view"""
        # print "View Iso!!"
        self.canva._display.View_Iso()
        self.canva._display.FitAll()

    def Select_All(self):
        """!Select all object in the viewer"""
        self.canva._display.selection = []
        self.canva._display.Context.EraseAll(True)
        for shape in self.canva._display.shapeLibrary.keys():
            for aFace in self.canva._display.shapeLibrary[shape].keys():
                self.canva._display.selection.append(aFace)
        for shape in self.canva._display.shapeLibrary.keys():
            self.canva._display.Draw(shape)

    def DeSelect_All(self):
        """!Deselect all object in the viewer"""
        self.canva._display.selection = []
        self.canva._display.Context.EraseAll(True)
        for shape in self.canva._display.shapeLibrary.keys():
            self.canva._display.Draw(shape)

    def Wireframe(self):
        """!Set the wireframe mode"""
        self.canva._display.SetModeWireFrame()

    def Shaded(self):
        """!Set shaded mode"""
        self.canva._display.SetModeShaded()

    def SetTogglesToFalse(self):
        """!Disable all toggles"""
        self.canva.Select = False
        self.canva.DynaZoom = False
        self.canva.DynaPan = False
        self.canva.DynaRotate = False

    def _refreshui(self):
        """!Refresh th GUI"""
        self.SetDynaCursor()

    def SetDynaCursor(self, iconfile=""):
        """!Set the cursor for zoom, pan or rotate."""
        pass
        # if iconfile:
        # img = wx.Bitmap(iconfile)
        # img_mask = wx.Mask(img, wx.Colour(255, 0, 255))
        # img.SetMask(img_mask)
        # cursor = wx.CursorFromImage(wx.ImageFromBitmap(img))
        # else:
        # cursor = wx.StockCursor(wx.CURSOR_DEFAULT)
        # self.SetCursor(cursor)

    def GetSelection(self, shape=None):
        """!Check if a shape is selected in the viewer."""
        return self.canva._display.GetSelection(shape)

def Grid(display):
    display.Viewer.Grid().SetColors(Quantity_Color(Quantity_NOC_GRAY), Quantity_Color(Quantity_NOC_YELLOW))
    display.Viewer.ActivateGrid(Aspect_GT_Rectangular, Aspect_GDM_Lines)
    display.Viewer.SetRectangularGridValues(0, 0, 10, 10, 0)
    display.Viewer.SetGridEcho(True)


def background():
    """!
    Locates the default background image.
    @return: absolute file name for the file default_background.bmp
    """
    bg_abs_filename = os.path.join('occ-data', 'background.bmp')
    if not os.path.isfile(bg_abs_filename):
        occ_package = sys.modules['OCC']
        bg_abs_filename = os.path.join(occ_package.__path__[0], 'Data', 'background.bmp')
        if not os.path.isfile(bg_abs_filename):
            raise NameError('Background image not found.')
    return bg_abs_filename


def icon(filename):
    """
    Gets the icon file path
    @param filename:
    @return: file path to icon
    """
    icon_abs_filename = os.path.join('occ-data', filename)
    if not os.path.isfile(icon_abs_filename):
        occ_package = sys.modules['OCC']
        icon_abs_filename = os.path.join(occ_package.__path__[0], 'Data', filename)
        if not os.path.isfile(icon_abs_filename):
            raise NameError('Icon not found.')
    return icon_abs_filename


###############################################################################    

def occtexture(data, filename):
    # Save the texture as a file
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from PIL import Image
    from OCC.Utils.Image import Texture
    dpi = 100.0
    errTex = array(data)
    cMin = errTex.min()
    cMax = errTex.max()
    w, h = errTex.shape[1] / dpi, errTex.shape[0] / dpi
    fig = plt.figure(figsize=(w, h), dpi=dpi)
    fig.figimage(errTex, cmap=cm.jet, vmin=cMin, vmax=cMax)
    if os.path.exists(filename):
        os.remove(filename)
    plt.savefig(filename)
    plt.close()
    im1 = Image.open(filename)
    im2 = im1.transpose(Image.FLIP_TOP_BOTTOM)
    im2.save(filename)
    tex = Texture(filename)

    return tex


###############################################################################
from PyQt5 import QtTest


def occwait(millisecs):
    QtTest.QTest.qWait(millisecs)

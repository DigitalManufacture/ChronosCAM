__author__ = "Anthony Beaucamp"
__copyright__ = "Copyright (C) 2022 Kyoto University"
__license__ = "Public Domain"
__version__ = "1.0"

from PyQt5.QtWidgets import QApplication, QFileDialog
from OCC.Display.backend import load_backend
load_backend("qt-pyqt5")

import importlib

from OCC.Tools.IO import *
from OCC.Tools.Viewer import *

class Window(MainWindow):

    def __init__(self, size=None, *args):
        MainWindow.__init__(self, size, *args)
        self.setWindowTitle("Chronos CAM - Build 2022/07/17")
        self.createMenu()
        self.createToolbar()
        self.plotting = False
        self.sections = []
        
    def call(self, func_name):
        module = importlib.import_module(func_name)
        importlib.reload(module)
        func = getattr(module, func_name.rpartition('.')[-1])
        func(self)

    def createMenu(self):
        self.add_menu("CAD")
        self.add_menu_item("CAD", "Import from STEP", self.importSTEP)
        self.add_menu_item("CAD", "Export to STEP", self.exportSTEP)
        self.add_menu_item("CAD", "Generate Ruled Geometry",      lambda:self.call('Functions.CAD.GenerateRuledGeometry')) 
        self.add_menu("CAM")
        self.add_menu_item("CAM", "Section Selected Faces",       lambda:self.call('Functions.CAM.SectionSelectedFaces'))
        self.add_menu_item("CAM", "Compute Principal Curvatures", lambda:self.call('Functions.CAM.ComputePrincipalCurvatures'))

    def createToolbar(self):
        self.add_tool_bar(os.path.join(self.getIcon("plot.png")), "Toggle Plotting", self.togglePlotting, toggable=True)        
        self.add_tool_bar(os.path.join(self.getIcon("erase.png")), "Erase All", self.eraseAll)

    def getIcon(self, filename):
        return os.path.join(os.getcwd(), "Icons", filename)
            
    def togglePlotting(self):
        self.plotting = not self.plotting
        
    def eraseAll(self):
        self.canva._display.Erase()    

    def importSTEP(self):
        fname = QFileDialog.getOpenFileName(self, 'Select CAD file to Import', os.getcwd(), 'CAD Files (*.stp)')[0]
        if fname:
            try:
                cad = occimport(fname)
                self.canva._display.Draw(cad[0], True)
                helpdlg(self, "File successfully imported.", "Import CAD")
            except:
                errordlg(self, "File cannot be imported.", "Import CAD")
                
    def exportSTEP(self):
        faces = self.GetSelection()                  
        if len(faces) < 1:
            warndlg(self, "Select surfaces for export.", "Export CAD")
            return
        shell = face2shell(faces)    
    
        fname = QFileDialog.getSaveFileName(self, 'Select CAD file to Export', os.getcwd(), 'CAD Files (*.stp)')[0]
        faces = self.GetSelection()                  
        if fname:
            try:
                cad = occexport(fname, [shell])
                helpdlg(self, "File successfully exported.", "Export CAD")
            except:
                errordlg(self, "File export failed.", "Export CAD")        
                

app = QApplication(sys.argv)
win = Window(size=[1600,900])
win.canva.InitDriver()
Grid(win.canva._display)
win.canva.qApp = app
win.canva._display.SetBackgroundImage(background())
win.show()
sys.exit(app.exec_())

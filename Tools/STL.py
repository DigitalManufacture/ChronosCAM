
__author__ = "Anthony Beaucamp"
__copyright__ = "Copyright (C) 2020 Kyoto University"
__license__ = "Public Domain"
__version__ = "1.0"

import os, sys
import datetime
from _winreg import *
import calendar

import sys, os, os.path
from numpy import *
from math import *
from math import *
from numpy import *
from numpy.linalg import *

from OCC.CTM import *


def stlimport(filepath):
    """!
    Imports from an STL file
    @param filepath: file location on disk
    @return: Mesh object
    """
    ctmOrigMesh = Mesh()
    offFilepath = filepath
    if ConvertAsciiStlToBinary(str(filepath), "samyakJain.stl"):
        OCCCTM_Import("samyakJain.stl", ctmOrigMesh)
        try:
            os.remove("samyakJain.stl")
        except:
            pass
    if NumVertices(ctmOrigMesh) == 0:
        # Read the mesh
        ctmOrigMesh = Mesh()
        # OCCCTM_Import(str(self.filepath), ctmMesh)
        OCCCTM_Import(str(filepath), ctmOrigMesh)
    return ctmOrigMesh


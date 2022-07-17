__author__ = "Anthony Beaucamp"
__copyright__ = "Copyright (C) 2022 Kyoto University"
__license__ = "Public Domain"
__version__ = "1.0"


def ComputePrincipalCurvatures(occ):
    
    if occ.sections == []:
        warndlg(occ, "Please generate sections first!", "Warning")
        return

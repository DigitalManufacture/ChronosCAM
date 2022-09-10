__author__ = "Anthony Beaucamp"
__copyright__ = "Copyright (C) 2022 Kyoto University"
__license__ = "Public Domain"
__version__ = "1.0"

# Implement "Matlab-Like" dialogs

from PyQt5.QtWidgets import QMessageBox

def helpdlg(win, msg, title):
    dialog = QMessageBox(win)
    dialog.setIcon(QMessageBox.Information)
    dialog.setText(msg)
    dialog.setWindowTitle(title)
    dialog.exec_()
    
def errordlg(win, msg, title):
    dialog = QMessageBox(win)
    dialog.setIcon(QMessageBox.Critical)
    dialog.setText(msg)
    dialog.setWindowTitle(title)
    dialog.exec_()    

def warndlg(win, msg, title):
    dialog = QMessageBox(win)
    dialog.setIcon(QMessageBox.Warning)
    dialog.setText(msg)
    dialog.setWindowTitle(title)
    dialog.exec_()  

def questdlg(win, msg, title):
    dialog = QMessageBox(win)
    dialog.setIcon(QMessageBox.Question)
    dialog.setText(msg)
    dialog.setWindowTitle(title)
    dialog.exec_()

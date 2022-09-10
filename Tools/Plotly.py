__author__ = "Shuntaro Yamato,Beaucamp Anthony"
__copyright__ = "Copyright (C) 2021 Kyoto University"
__license__ = "Public Domain"
__version__ = "1.0"


# %% Initialization
def initialize():
    global figcount, figall, winall, fig, showPlot
    figcount = 0
    figall = []
    winall = []
    fig = []
    showPlot = []


# %% Import modules and initialization
from screeninfo import get_monitors  # 20210621 added

for gca in get_monitors():  # 20210621 added
    pass  # 20210621 added
initialize()

# "Matlab" style functions for plotting
import numpy as np

# %% Define global parameter in module
global fig
global figall
global winall
global figcount  # 20210621追加
global figcol  # 20210621追加
global figrow  # 20210621追加
global showPlot
global subplotidx
figcount = 0
figall = []
winall = []
fig = []
showPlot = []
# figcount = 0 # 20210621追加
# figcol = 1 # 20210621追加
# figrow = 1

# %% Plotlyviewer
import os, sys

from PyQt5 import QtCore, QtWidgets

import ctypes

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
import plotly.graph_objects as go
import plotly.offline
from PyQt5.QtWidgets import QApplication
from win32gui import SetWindowPos
import win32con


class PlotlyViewer(QtWebEngineWidgets.QWebEngineView):
    def __init__(self, fig, config=None, exec=False):
        # Create a QApplication instance or use the existing one if it exists
        app = QApplication.instance()
        if not app:  # create QApplication if it doesnt exist
            app = QApplication(sys.argv)

        super().__init__()
        self.config = {'displaylogo': False}
        if config:
            self.config[config[0]] = config[1]
        self.createWindow(QtWebEngineWidgets.QWebEnginePage.WebBrowserWindow)
        # self.setWindowTitle(str(fig.layout.title.text)) # "Plotly Viewer"
        self.setWindowTitle("Figure " + str(figcount))  # 図の順番が分かりやすいためこっち採用
        # print(fig.layout.title.text)
        filename = "TempFig" + str(figcount) + ".html"  # 参照するだけならglobal宣言はいらない
        # このfilenameが変わるようにしないと図が上書きされる
        self.file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))
        # print(self.file_path)
        if fig.layout.width is not None and fig.layout.height is not None:
            self.resize(fig.layout.width + 30, fig.layout.height + 30)
        self.update(fig)
        SetWindowPos(self.winId(), win32con.HWND_TOPMOST,  # = always on top. only reliable way to bring it to the front on windows
                     0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW)
        SetWindowPos(self.winId(), win32con.HWND_NOTOPMOST,  # disable the always on top, but leave window at its top position
                     0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW)
        self.raise_()
        self.show()
        self.activateWindow()

        if exec:
            app.exec_()

    def update(self, fig):
        plotly.offline.plot(fig, filename=self.file_path, auto_open=False, config=self.config)
        self.load(QtCore.QUrl.fromLocalFile(self.file_path))

    def closeEvent(self, event):
        os.remove(self.file_path)


# %% Color def
coldef = {'r': 'rgba(200,0,0,1)',
          'g': 'rgba(0,200,0,1)',
          'b': 'rgba(0,0,200,1)',
          'k': 'rgba(0,0,0,1)',
          'lime': 'rgba(144,255,59,1)',
          'lightpink': 'rgba(255,182,193,1)',
          'c': 'rgba(0,255,255,1)',
          'w': 'rgba(255,255,255,1)',
          }

# %% Marker symbol def
marksymdef = {'o': 'circle',
              's': 'square',
              'd': 'diamond',
              '+': 'cross',
              'X': 'x',
              'x': 'x-thin',
              '^': 'triangle-up',
              'v': 'triangle-down',
              '<': 'triangle-left',
              '>': 'triangle-right',
              'p': 'pentagon',
              'h': 'hexagon',
              '*': 'asterisk',
              'Y': 'y-down'
              }

# %% Line style def
linestyledef = {'-': None,
                '--': 'dash',
                ':': 'dot',
                '-.': 'dashdot'
                }

# %% Colormap style def
import plotly.express as px

maps = px.colors.sequential.swatches_continuous()
colmapdef = {'jet': ['rgb(0,0,131)', 'rgb(0,60,170)', 'rgb(5,255,255)', 'rgb(255,255,0)', 'rgb(250,0,0)', 'rgb(128,0,0)']
             }

# %% Stop command (corresponding to return command in MATLAB)
"""
def stop():
    import pdb
    pdb.set_trace()
"""


# %% Display figure
# bgcolor = 'rgb(243, 243, 243)'
def figure(size=None, title=None, font="Arial", fontsize=12, bgcolor=None, plbgcolor=None, pabgcolor=None,
           fontcolor='k', autosize=True, margin=None, config=None, visible=True):
    # Prepare plot description
    global win, figall, figcount, fig
    figcount = figcount + 1

    # DefaultでFigure numberをtitleとして付けるとき
    """
    if title is None:
        title = 'Figure ' + str(figcount)
        # print(title)
    else:
        pass
    """
    if fontcolor in coldef:
        fontcolor = coldef[fontcolor]
    else:
        pass

    fontspec = {"family": font, "size": fontsize, "color": fontcolor}
    try:
        scale = float(ctypes.windll.shcore.GetScaleFactorForDevice(0)) / 100
    except:
        scale = 0.666
    if size is None:
        layout = go.Layout(title=title, font=fontspec)
    else:
        layout = go.Layout(height=size[1]/scale, width=size[0]/scale, title=title, font=fontspec)

        # Treatment for background colors
    if bgcolor is not None:
        if bgcolor in coldef:
            bgcolor = coldef[bgcolor]
        else:
            pass
    if plbgcolor is not None:
        if plbgcolor in coldef:
            plbgcolor = coldef[plbgcolor]
        else:
            pass
    if plbgcolor is None and bgcolor is not None:
        plbgcolor = bgcolor

    if pabgcolor is not None:
        if pabgcolor in coldef:
            pabgcolor = coldef[pabgcolor]
        elif pabgcolor == 'same':
            pabgcolor = plbgcolor
        else:
            pass

    if margin is not None:
        for i in range(4):
            if np.size(margin[i]) == 0:
                margin[i] = None
            else:
                pass
        margin = {'l': margin[0], 'r': margin[1], 't': margin[2], 'b': margin[3]}
    else:
        margin = {'l': 10, 'r': 0, 't': 40, 'b': 10}
        # margin = {'l':20,'r':20,'t':25,'b':25}

    fig = go.Figure(layout=layout)
    fig.update_layout(showlegend=False, autosize=autosize, margin=margin,
                      plot_bgcolor=plbgcolor, paper_bgcolor=pabgcolor)

    fig.update_xaxes(showgrid=False, zeroline=False, showline=True)
    fig.update_yaxes(showgrid=False, zeroline=False, showline=True)
    figall.append(fig)
    showPlot.append(visible)
    if visible:
        win = PlotlyViewer(fig, config=config)
        winall.append(win)
        
    fig.update_layout(width=None, height=None, autosize=True)


# %% Subplot function
from plotly.subplots import make_subplots


def subplot(grid, specs='2D', xshare=False, yshare=False):
    global winall, figall, figrow, figcol, subplotidx  # figcol, figrow-> 20210621 added

    # 20210621 added
    # """
    figrow = (grid[2] + (grid[1] - 1)) // grid[1]
    figcol = grid[2] - (figrow - 1) * grid[1]
    subplotidx = grid[2]
    # print(figrow)
    # print(figcol)
    # """

    if not any(figall):  # if there are no figure frame
        print('No figure')
        return
    if figall[figcount-1]._has_subplots():  # if subplots exist
        return [figrow, figcol]  # 20210621 added
    else:
        if specs == '2D':
            specs = []
            for i in range(grid[0]):
                spec = []  # Specsとspecの使い分け注意
                for j in range(grid[1]):
                    spec.append({'type': 'xy'})  # Default: 2D plot
                specs.append(spec)  # インデント注意
        elif specs == '3D':
            specs = []
            for i in range(grid[0]):
                spec = []  # Specsとspecの使い分け注意
                for j in range(grid[1]):
                    spec.append({'type': 'scene'})  # Default: 2D plot
                specs.append(spec)  # インデント注意
                # print(specs)

        figall[figcount-1].set_subplots(rows=grid[0], cols=grid[1], specs=specs, shared_xaxes=xshare, shared_yaxes=yshare)
        figall[figcount-1].update_layout(showlegend=False)
        figall[figcount-1].update_xaxes(showgrid=False, zeroline=False)
        figall[figcount-1].update_yaxes(showgrid=False, zeroline=False)
        if showPlot[figcount-1]:
            winall[figcount-1].update(figall[figcount-1])

        return [figrow, figcol]

    # fig.layout.height = fig.layout.height / grid[0]
    # fig.layout.height = fig.layout.width / grid[1]
    # fig = make_subplots(rows=grid[0], cols=grid[1])
    # fig.update_layout(showlegend=False, title_text="Specs with Subplot Title")
    # return fig


# %% Grid off function
"""
def gridoff(ax=None):
    global winall, figall, figcount, figrow, figcol
    if figall[figcount-1]._has_subplots(): # if subplots exist
        if ax is None:
            figall[figcount-1].update_xaxes(showgrid=False, zeroline=False,row=figrow,col=figcol) 
            figall[figcount-1].update_yaxes(showgrid=False, zeroline=False,row=figrow,col=figcol)
        else:
            figall[figcount-1].update_xaxes(showgrid=False, zeroline=False,row=ax[0],col=ax[1]) 
            figall[figcount-1].update_yaxes(showgrid=False, zeroline=False,row=ax[0],col=ax[1])

    else:
        figall[figcount-1].update_xaxes(showgrid=False, zeroline=False) 
        figall[figcount-1].update_yaxes(showgrid=False, zeroline=False)
    winall[figcount-1].update(figall[figcount-1])
"""


# %% General grid function
def grid(mode, axis='xy', ax=None, color=None, width=None):
    global winall, figall, figcount, figrow, figcol

    if not any(figall):  # if there are no figure frame
        print('No figure')
        return

    if figall[figcount-1]._has_subplots():  # if subplots exist
        if ax is None:  # no grid designation
            row = figrow
            col = figcol
        else:
            row = ax[0]
            col = ax[1]
    else:
        row = 1
        col = 1

    if color is not None:
        if color in coldef:
            color = coldef[color]
        else:
            pass

    # control properties
    if mode == 'on':
        if figall[figcount-1]._has_subplots():
            if axis == 'xy':
                figall[figcount-1].update_xaxes(showgrid=True, zeroline=False, gridcolor=color, gridwidth=width,
                                                  row=row, col=col)
                figall[figcount-1].update_yaxes(showgrid=True, zeroline=False, gridcolor=color, gridwidth=width,
                                                  row=row, col=col)
            elif axis == 'x':
                figall[figcount-1].update_xaxes(showgrid=True, zeroline=False, gridcolor=color, gridwidth=width,
                                                  row=row, col=col)
            elif axis == 'y':
                figall[figcount-1].update_yaxes(showgrid=True, zeroline=False, gridcolor=color, gridwidth=width,
                                                  row=row, col=col)
        else:
            if axis == 'xy':
                figall[figcount-1].update_xaxes(showgrid=True, zeroline=False, gridcolor=color, gridwidth=width,
                                                  )
                figall[figcount-1].update_yaxes(showgrid=True, zeroline=False, gridcolor=color, gridwidth=width,
                                                  )
            elif axis == 'x':
                figall[figcount-1].update_xaxes(showgrid=True, zeroline=False, gridcolor=color, gridwidth=width,
                                                  )
            elif axis == 'y':
                figall[figcount-1].update_xaxes(showgrid=True, zeroline=False, gridcolor=color, gridwidth=width,
                                                  )
    else:
        if figall[figcount-1]._has_subplots():
            if axis == 'xy':
                figall[figcount-1].update_xaxes(showgrid=False, row=row, col=col)
                figall[figcount-1].update_yaxes(showgrid=False, row=row, col=col)
            elif axis == 'x':
                figall[figcount-1].update_xaxes(showgrid=False, row=row, col=col)
            elif axis == 'y':
                figall[figcount-1].update_yaxes(showgrid=False, row=row, col=col)
        else:
            if axis == 'xy':
                figall[figcount-1].update_xaxes(showgrid=False)
                figall[figcount-1].update_yaxes(showgrid=False)
            elif axis == 'x':
                figall[figcount-1].update_xaxes(showgrid=False)
            elif axis == 'y':
                figall[figcount-1].update_yaxes(showgrid=False)

    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% Grid on function
def gridon(ax=None, color=None, width=None):
    global winall, figall, figcount, figrow, figcol

    if not any(figall):  # if there are no figure frame
        print('No figure')
        return

    if figall[figcount-1]._has_subplots():  # if subplots exist
        if ax is None:  # no grid designation
            row = figrow
            col = figcol
        else:
            row = ax[0]
            col = ax[1]

    if color is not None:
        if color in coldef:
            color = coldef[color]
        else:
            pass

    # control properties
    prop = dict(showgrid=True, zeroline=False, gridcolor=color, gridwidth=width)

    if figall[figcount-1]._has_subplots():  # if subplots exist
        figall[figcount-1].update_layout(xaxis=prop, yaxis=prop, row=row, col=col)
    else:
        figall[figcount-1].update_layout(xaxis=prop, yaxis=prop)

    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% log plot function
def logplot(axis, ax=None):
    global winall, figall, figcount, figrow, figcol

    if not any(figall):  # if there are no figure frame
        print('No figure')
        return

    if figall[figcount-1]._has_subplots():  # if subplots exist
        if ax is None:  # no grid designation
            row = figrow
            col = figcol
        else:
            row = ax[0]
            col = ax[1]

    if axis == 'x':
        xaxis = dict(type='log')
        if figall[figcount-1]._has_subplots():  # if subplots exist
            figall[figcount-1].update_layout(xaxis=xaxis, row=row, col=col)
        else:
            figall[figcount-1].update_layout(xaxis=xaxis)
    elif axis == 'y':
        yaxis = dict(type='log')
        if figall[figcount-1]._has_subplots():  # if subplots exist
            figall[figcount-1].update_layout(yaxis=yaxis, row=row, col=col)
        else:
            figall[figcount-1].update_layout(yaxis=yaxis)
    elif axis == 'xy':
        xaxis = dict(type='log')
        yaxis = dict(type='log')
        if figall[figcount-1]._has_subplots():  # if subplots exist
            figall[figcount-1].update_layout(xaxis=xaxis, yaxis=yaxis, row=row, col=col)
        else:
            figall[figcount-1].update_layout(xaxis=xaxis, yaxis=yaxis)

    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% X label function
def labels(x=None, y=None, z=None, ax=None, font=None, fontsize=None, fontcolor=None,
           bgcolor=None):
    global winall, figall, figcount, figrow, figcol, subplotidx

    if not any(figall):  # if there are no figure frame
        print('No figure')
        return

    if font is None:
        font = figall[figcount-1].layout.font.family
    if fontsize is None:
        fontsize = figall[figcount-1].layout.font.size
    if fontcolor is None:
        fontcolor = figall[figcount-1].layout.font.color
    else:
        if fontcolor in coldef:
            fontcolor = coldef[fontcolor]
        else:
            pass
    if bgcolor in coldef:
        bgcolor = coldef[bgcolor]
    else:
        pass

    fontspec = {"family": font, "size": fontsize, "color": fontcolor}

    if figall[figcount-1]._has_subplots():  # if subplots exist
        idx = subplotidx
        if ax is None:  # no grid designation
            row = figrow
            col = figcol
        else:
            row = ax[0]
            col = ax[1]
    else:
        idx = 1

    if len(x) == 0:
        x = None
    if len(y) == 0:
        y = None

    if z is None:  # 2D label
        if figall[figcount-1]._has_subplots():  # if subplots exist
            figall[figcount-1].update_xaxes(title=x, title_font=fontspec, row=row, col=col)
            figall[figcount-1].update_yaxes(title=y, title_font=fontspec, row=row, col=col)
        else:
            figall[figcount-1].update_xaxes(title=x, title_font=fontspec)
            figall[figcount-1].update_yaxes(title=y, title_font=fontspec)

    else:
        if len(z) == 0:
            z = None

        xaxis = dict(title=x, title_font=fontspec)
        yaxis = dict(title=y, title_font=fontspec)
        zaxis = dict(title=z, title_font=fontspec)

        idx = 'scene' + str(idx)
        figall[figcount-1].layout[idx].xaxis.update(xaxis)
        figall[figcount-1].layout[idx].yaxis.update(yaxis)
        figall[figcount-1].layout[idx].zaxis.update(zaxis)
        figall[figcount-1].layout[idx].bgcolor = bgcolor

    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% Indivisual y label function
def ylabel(title, ax=None, fontsize=None, font=None, fontcolor=None, bgcolor=None):
    global winall, figall, figcount, figrow, figcol

    if not any(figall):  # if there are no figure frame
        print('No figure')
        return

    if figall[figcount-1]._has_subplots():  # if subplots exist
        if ax is None:  # no grid designation
            row = figrow
            col = figcol
        else:
            row = ax[0]
            col = ax[1]

    if font is None:
        font = figall[figcount-1].layout.font.family
    if fontsize is None:
        fontsize = figall[figcount-1].layout.font.size
    if fontcolor is None:
        fontcolor = figall[figcount-1].layout.font.color
    else:
        if fontcolor in coldef:
            fontcolor = coldef[fontcolor]
        else:
            pass

    if bgcolor in coldef:
        bgcolor = coldef[bgcolor]
    else:
        pass

    fontspec = {"family": font, "size": fontsize, "color": fontcolor}

    prop = dict(title=title, titlefont=fontspec)

    if figall[figcount-1]._has_subplots():  # if subplots exist
        figall[figcount-1].update_layout(yaxis=prop, row=row, col=col)
    else:
        figall[figcount-1].update_layout(yaxis=prop)

    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% xticks function
def xticks(tilt=None, lim=None, ax=None):
    global winall, figall, figcount, figrow, figcol

    if not any(figall):  # if there are no figure frame
        print('No figure')
        return

    if figall[figcount-1]._has_subplots():  # if subplots exist
        if ax is None:  # no grid designation
            row = figrow
            col = figcol
        else:
            row = ax[0]
            col = ax[1]

    prop = dict(tickangle=tilt, range=lim)

    if figall[figcount-1]._has_subplots():  # if subplots exist
        figall[figcount-1].update_layout(xaxis=prop, row=row, col=col)
    else:
        figall[figcount-1].update_layout(xaxis=prop)

    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% xlim function
def xlim(lim, ax=None):
    global winall, figall, figcount, figrow, figcol

    if not any(figall):  # if there are no figure frame
        print('No figure')
        return

    if figall[figcount-1]._has_subplots():  # if subplots exist
        if ax is None:  # no grid designation
            row = figrow
            col = figcol
        else:
            row = ax[0]
            col = ax[1]

    prop = dict(range=lim)

    if figall[figcount-1]._has_subplots():  # if subplots exist
        figall[figcount-1].update_layout(xaxis=prop, row=row, col=col)
    else:
        figall[figcount-1].update_layout(xaxis=prop)

    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% zlim function
def zlim(lim):
    global winall, figall, figcount, figrow, figcol, subplotidx

    if not any(figall):  # if there are no figure frame
        print('No figure')
        return

    if figall[figcount-1]._has_subplots():  # if subplots exist
        idx = subplotidx
    else:
        idx = 1

    zaxis = dict(range=lim)

    idx = 'scene' + str(idx)
    figall[figcount-1].layout[idx].zaxis.update(zaxis)

    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% xticks function
def yticks(tilt=None, lim=None, ax=None):
    global winall, figall, figcount, figrow, figcol

    if not any(figall):  # if there are no figure frame
        print('No figure')
        return

    if figall[figcount-1]._has_subplots():  # if subplots exist
        if ax is None:  # no grid designation
            row = figrow
            col = figcol
        else:
            row = ax[0]
            col = ax[1]

    prop = dict(tickangle=tilt, range=lim)

    if figall[figcount-1]._has_subplots():  # if subplots exist
        figall[figcount-1].update_layout(yaxis=prop, row=row, col=col)
    else:
        figall[figcount-1].update_layout(yaxis=prop)

    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% xlim function
def ylim(lim, ax=None):
    global winall, figall, figcount, figrow, figcol

    if not any(figall):  # if there are no figure frame
        print('No figure')
        return

    if figall[figcount-1]._has_subplots():  # if subplots exist
        if ax is None:  # no grid designation
            row = figrow
            col = figcol
        else:
            row = ax[0]
            col = ax[1]

    prop = dict(range=lim)

    if figall[figcount-1]._has_subplots():  # if subplots exist
        figall[figcount-1].update_layout(yaxis=prop, row=row, col=col)
    else:
        figall[figcount-1].update_layout(yaxis=prop)

    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% Z label function (need to be fixed)
"""
def zlabel(zLabel,ax=None,font=None,fontsize=None,fontcolor=None):
    global winall, figall, figcount, figrow, figcol

    if font is None:
        font = figall[figcount-1].layout.font.family
    if fontsize is None:
        fontsize = figall[figcount-1].layout.font.size
    if fontcolor is None:
        fontcolor = figall[figcount-1].layout.font.color
    else:
        fontcolor = coldef[fontcolor]
    
    fontspec = {"family":font,"size":fontsize,"color":fontcolor}
    if figall[figcount-1]._has_subplots(): # if subplots exist
        if ax is None:
            figall[figcount-1].update_zaxes(title=zLabel,title_font=fontspec,row=figrow,col=figcol) 
        else:
            figall[figcount-1].update_zaxes(title=zLabel,title_font=fontspec,color=coldef['g'],row=ax[0],col=ax[1]) 
    else:
        figall[figcount-1].update_zaxes(title=zLabel,title_font=fontspec) 
    winall[figcount-1].update(figall[figcount-1])
"""


# %% Legend function
def legend(ax=None, color=None, fontsize=None, order=None, position=None, bgcolor=None, bordercolor=None):
    global winall, figall, figcount, figrow, figcol

    if figall[figcount-1]._has_subplots():  # if subplots exist
        if ax is None:  # no grid designation
            row = figrow
            col = figcol
        else:
            row = ax[0]
            col = ax[1]

    if position is not None:
        if position == 'northwestin':
            x = 0
            y = 1.0
        else:
            x = position[0]
            y = position[1]
    else:
        x = None
        y = None

    if bgcolor is not None:
        if bgcolor in coldef:
            bgcolor = coldef[bgcolor]
        else:
            pass

    if bordercolor is not None:
        if bordercolor in coldef:
            bordercolor = coldef[bordercolor]
        else:
            pass

    # control properties
    # prop = dict(showlegend=True,text=label)
    prop = dict(x=x, y=y, traceorder=order, font_size=fontsize, bgcolor=bgcolor, )

    if figall[figcount-1]._has_subplots():  # if subplots exist
        figall[figcount-1].update_layout(showlegend=True, legend=prop)
    else:
        figall[figcount-1].update_layout(showlegend=True, legend=prop)

    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% hover function
def hover(hoverinfo, ax=None):
    global winall, figall, figcount, figrow, figcol

    if figall[figcount-1]._has_subplots():  # if subplots exist
        if ax is None:  # no grid designation
            row = figrow
            col = figcol
        else:
            row = ax[0]
            col = ax[1]

    if figall[figcount-1]._has_subplots():  # if subplots exist
        figall[figcount-1].update_traces(hoverinfo=hoverinfo, row=row, col=col)
    else:
        figall[figcount-1].update_traces(hoverinfo=hoverinfo)

    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% Control view points
def view(up=[0, 0, 1], center=[0, 0, 0], eye=[-1.25, -1.25, 1.25], ax=None, dragmode=None, orthographic=False):
    global winall, figall, figcount, figrow, figcol, subplotidx

    camera = dict(up=dict(x=up[0], y=up[1], z=up[2]), center=dict(x=center[0], y=center[1], z=center[2]),
                  eye=dict(x=eye[0], y=eye[1], z=eye[2]))

    if figall[figcount-1]._has_subplots():  # if subplots exist
        idx = subplotidx
    else:
        idx = 1
    idx = 'scene' + str(idx)
    figall[figcount-1].layout[idx].camera = camera
    if orthographic:
        figall[figcount-1].layout[idx].camera.projection.type = "orthographic"
    figall[figcount-1].layout[idx].dragmode = dragmode

    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% Control view points
"""
def view(v,ax=None):
    global winall, figall, figcount, figrow, figcol, subplotidx
    
    # camera_eye=dict(x=v[0]*1.25, y=v[1]*1.25, z=v[2]*1.25)
    camera_eye=dict(x=v[0], y=v[1], z=v[2])
    
    if figall[figcount-1]._has_subplots(): # if subplots exist
        idx = subplotidx
    else:
        idx = 1
    idx = 'scene' + str(idx)
    figall[figcount-1].layout[idx].camera.eye = camera_eye

    winall[figcount-1].update(figall[figcount-1])
"""


# %% control figure aspect ratio
def pdaspect(v=None, ax=None, mode=None):
    global winall, figall, figcount, figrow, figcol

    if mode is not None:
        scene = dict(aspectmode=mode)
    else:
        # camera_eye=dict(x=v[0]*1.25, y=v[1]*1.25, z=v[2]*1.25)
        aspect = dict(x=v[0], y=v[1], z=v[2])
        scene = dict(aspectratio=aspect)

    if figall[figcount-1]._has_subplots():  # if subplots exist
        if ax is None:
            figall[figcount-1].update_layout(scene=scene, row=figrow, col=figcol)
        else:
            figall[figcount-1].update_layout(scene=scene, row=ax[0], col=ax[1])
    else:
        figall[figcount-1].update_layout(scene=scene)

    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% control figure aspect ratio
def showscale(mode, ax=None):
    global winall, figall, figcount, figrow, figcol

    if mode == 'off':
        showscale = False
    else:
        showscale = True

    if figall[figcount-1]._has_subplots():  # if subplots exist
        if ax is None:  # no grid designation
            row = figrow
            col = figcol
        else:
            row = ax[0]
            col = ax[1]

    if figall[figcount-1]._has_subplots():  # if subplots exist
        figall[figcount-1].update_traces(showscale=showscale, row=row, col=col)
    else:
        figall[figcount-1].update_traces(showscale=showscale)

    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% categoryorderdef ####################################################
categoryorderdef = {'ascend': 'category ascending',
                    'descend': 'category ascending',
                    'yascend': 'total ascending',
                    'ydescend': 'total descending',
                    }


# %% Bar mode function 
def barmode(mode=None, ax=None, bargap=None, groupgap=None, xorder=None):
    global winall, figall, figcount, figcol, figrow
    if not any(figall):  # if there are no figure frame
        figure()

    if figall[figcount-1]._has_subplots():  # if subplots exist
        if ax is None:  # no grid designation
            row = figrow
            col = figcol
        else:
            row = ax[0]
            col = ax[1]

    if xorder is not None:
        if type(xorder) is list:
            categoryorder = 'array'
            categoryarray = xorder
        else:
            if xorder in categoryorderdef:
                categoryorder = categoryorderdef[xorder]
            else:
                categoryorder = xorder
            categoryarray = None
    else:
        categoryorder = None
        categoryarray = None
    xaxis = dict(categoryorder=categoryorder, categoryarray=categoryarray)

    if figall[figcount-1]._has_subplots():  # if subplots exist
        figall[figcount-1].update_layout(barmode=mode, bargap=bargap, bargroupgap=groupgap,
                                           xaxis=xaxis, row=row, col=col)
    else:
        figall[figcount-1].update_layout(barmode=mode, bargap=bargap, bargroupgap=groupgap, xaxis=xaxis)

        # Update figure
    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% 2D charts functions
# %% 2D plot function
def plot(x, y, style=None, ax=None, color=None, linecolor=None, linewidth=None, dataname=None,
         connectgap=False, interp='linear', hover=None, hoverinfo=None, showfacecolor=True,
         markersize=None, markerfacecolor=None, markeredgecolor=None, markeredgewidth=None):
    global winall, figall, figcount, figcol, figrow
    if not any(figall):  # if there are no figure frame
        figure()

    x = np.array(x)
    y = np.array(y)
    # Generate x data if it's empty
    if np.size(x) == 0:
        x = np.linspace(0, np.size(y), np.size(y))
        # 点が1つしかないとscatterがエラーをおこすため，応急的な処置->良い方法ない？
    if np.size(x) == 1 and np.size(y) == 1:
        x = [x]
        y = [y]
        if style is None:
            style = 'o'

    if style is None:
        style = '-'

    # Detect line/marker style
    if style[0] in linestyledef:  # exist line style indicator
        mode = 'lines+markers'
        if style[0] == style[:]:  # '-' or ':'
            linestyle = linestyledef[style[0]]
            markerstyle = style.replace(style[0], '')
        else:  # other indicator exist
            if style[0:2] in linestyledef:  # exist second line style indicator
                linestyle = linestyledef[style[0:2]]
                markerstyle = style.replace(style[0:2], '')
            else:
                linestyle = linestyledef[style[0]]
                markerstyle = style.replace(style[0], '')
    else:  # not exist line-style indicator
        mode = 'markers'
        linestyle = None
        markerstyle = style
    # print(linestyle)

    if len(markerstyle) == 0:
        mode = 'lines'
        markerstyle = None

    if markerstyle is not None:
        if markerstyle[-1] == '.':  # exist dot indicator
            if showfacecolor is False:  # when markerfacecolor is off
                markerfacecolor = None  # 後の処理の為にNoneにしておく
                markerstyle = marksymdef[markerstyle[:-1]] + '-open-dot'
            else:  # when markerfacecolor is on
                markerstyle = marksymdef[markerstyle[:-1]] + '-dot'
        else:  # not exist dot indicator
            if showfacecolor == False:  # when markerfacecolor is off
                markerfacecolor = None  # 後の処理の為にNoneにしておく
                markerstyle = marksymdef[markerstyle] + '-open'
            else:  # when markerfacecolor is on
                markerstyle = marksymdef[markerstyle]

    # Detect line color
    if color is not None:
        if color in coldef:
            color = coldef[color]
        else:
            pass
    if linecolor is not None:
        if linecolor in coldef:
            linecolor = coldef[linecolor]
        else:
            pass
    if linecolor is None and color is not None:
        linecolor = color

    # Detect marker-face color
    if markerfacecolor is None:
        markerfacecolor = linecolor
    else:
        if markerfacecolor in coldef:
            markerfacecolor = coldef[markerfacecolor]
        else:
            pass

            # Detect marker-edge color
    if markeredgecolor is None:
        markeredgecolor = linecolor
    else:
        if markeredgecolor in coldef:
            markeredgecolor = coldef[markeredgecolor]
        else:
            pass

            # hover control
    if hover is not None:
        # hoverinfo = 'text+name'
        pass

    # Control line/marker properties
    if mode == 'lines':
        line = dict(color=linecolor, width=linewidth, dash=linestyle, shape=interp)
    elif mode == 'markers':
        marker = dict(symbol=markerstyle, color=markerfacecolor, size=markersize,
                      line=dict(width=markeredgewidth, color=markeredgecolor))
    else:
        line = dict(color=linecolor, width=linewidth, dash=linestyle, shape=interp)
        marker = dict(symbol=markerstyle, color=markerfacecolor, size=markersize,
                      line=dict(width=markeredgewidth, color=markeredgecolor))

    if figall[figcount-1]._has_subplots():  # if subplots exist
        if ax is None:  # no grid designation
            row = figrow
            col = figcol
        else:
            row = ax[0]
            col = ax[1]

    if figall[figcount-1]._has_subplots():  # if subplots exist
        if mode == 'lines':
            figall[figcount-1].add_trace(go.Scatter(x=x, y=y, mode=mode, line=line, name=dataname
                                                      ), row=row, col=col)
        elif mode == 'markers':
            figall[figcount-1].add_trace(go.Scatter(x=x, y=y, mode=mode, marker=marker, name=dataname),
                                           row=row, col=col)
        else:
            figall[figcount-1].add_trace(go.Scatter(x=x, y=y, mode=mode, line=line, marker=marker,
                                                      name=dataname
                                                      ), row=row, col=col
                                           )
    else:
        if mode == 'lines':
            figall[figcount-1].add_trace(go.Scatter(x=x, y=y, mode=mode, line=line, name=dataname,
                                                      connectgaps=connectgap, )
                                           )
        elif mode == 'markers':
            figall[figcount-1].add_trace(go.Scatter(x=x, y=y, mode=mode, marker=marker, name=dataname,
                                                      )
                                           )
        else:
            figall[figcount-1].add_trace(go.Scatter(x=x, y=y, mode=mode, line=line, marker=marker, name=dataname,
                                                      connectgaps=connectgap, text=hover, hoverinfo=hoverinfo),
                                           )

    # Update figure
    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% fill function
def fill(x, y, ax=None, color=None, style='toself', dataname=None, edgecolor='rgba(255,255,255,0)',
         showlegend=False, alpha=None, edgewidth=None, edgestyle='-'):
    global winall, figall, figcount, figcol, figrow
    if not any(figall):  # if there are no figure frame
        print('No figure')
        return

    if color in coldef:
        color = coldef[color]
    else:
        pass

    if edgecolor in coldef:
        edgecolor = coldef[edgecolor]
    else:
        pass

    if figall[figcount-1]._has_subplots():  # if subplots exist
        if ax is None:  # no grid designation
            row = figrow
            col = figcol
        else:
            row = ax[0]
            col = ax[1]

    edgestyle = linestyledef[edgestyle]

    line = dict(color=edgecolor, width=edgewidth, dash=edgestyle)

    if figall[figcount-1]._has_subplots():  # if subplots exist
        figall[figcount-1].add_trace(go.Scatter(x=x, y=y, fill=style, fillcolor=color, line=line,
                                                  showlegend=False, opacity=alpha), row=row, col=col
                                       )
    else:
        figall[figcount-1].add_trace(go.Scatter(x=x, y=y, fill=style, fillcolor=color, line=line,
                                                  showlegend=False, opacity=alpha)
                                       )

    # Update figure
    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% Scatter function (plot function includes scatter function)
def scatter(x, y, ax=None, facecolor=None, style='o', size=None, markersize=None, alpha=None, hover=None,
            sizemode=None, edgecolor=None, edgewidth=None, colormap=None,
            showscale=False, sizeref=None, dataname=None, showfacecolor=True):
    global winall, figall, figcount, figcol, figrow
    if not any(figall):  # if there are no figure frame
        figure()

    x = np.array(x)
    y = np.array(y)
    # Generate x data if it's empty
    if np.size(x) == 0:
        x = np.linspace(0, np.size(y), np.size(y))
        # 点が1つしかないとscatterがエラーをおこすため，応急的な処置->良い方法ない？
    if np.size(x) == 1 and np.size(y) == 1:
        x = [x]
        y = [y]

    # marker style treatment
    if style[-1] == '.':  # 後ろがdotの場合
        if showfacecolor is False:  # dot指定があり，facecolorがoffの場合
            style = marksymdef[style[:-1]] + '-open-dot'
        else:  # dot指定があり,facecolorはon
            style = marksymdef[style[:-1]] + '-dot'
    else:  # 後ろがdotではない場合
        if showfacecolor is False:  # dot指定無しで，facecolorがoffの場合
            style = marksymdef[style] + '-open'
        else:  # dot指定なしで，facecolorはon
            style = marksymdef[style]

    # facecolor treatment
    temp = []
    if facecolor is not None:
        for i in range(len(facecolor)):
            if facecolor[i] in coldef:
                temp = np.append(temp, coldef[facecolor[i]])
            else:
                temp = np.append(temp, facecolor[i])
        facecolor = temp
    else:
        pass

    if facecolor is not None:
        if len(facecolor) == 1:
            temp = facecolor
            for i in range(len(x)):
                facecolor = np.append(facecolor, temp)
        else:
            pass

    # Colormap treatment when intensity is designated
    if colormap is not None:
        showscale = True

    if sizeref is not None:
        sizemode = 'area'

    if markersize is not None and size is None:
        size = markersize

    # Define marker propoeties
    marker = dict(symbol=style, color=facecolor, size=size, opacity=alpha,
                  showscale=showscale, colorscale=colormap, sizeref=sizeref, sizemode=sizemode,
                  line=dict(width=edgewidth, color=edgecolor))

    if figall[figcount-1]._has_subplots():  # if subplots exist
        if ax is None:  # no grid designation
            row = figrow
            col = figcol
        else:
            row = ax[0]
            col = ax[1]

    if figall[figcount-1]._has_subplots():  # if subplots exist
        figall[figcount-1].add_trace(go.Scatter(x=x, y=y, mode='markers', marker=marker,
                                                  text=hover, name=dataname,
                                                  ), row=row, col=col
                                       )
    else:
        figall[figcount-1].add_trace(go.Scatter(x=x, y=y, mode='markers', marker=marker,
                                                  text=hover, name=dataname,
                                                  )
                                       )

    # Update figure
    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% Scattergl function for large dataset
def scattergl(x, y, ax=None, facecolor=None, style='o', size=None, markersize=None, alpha=None, hover=None,
              sizemode=None, edgecolor=None, edgewidth=None, colormap=None,
              showscale=False, sizeref=None, dataname=None, showfacecolor=True):
    global winall, figall, figcount, figcol, figrow
    if not any(figall):  # if there are no figure frame
        figure()

    x = np.array(x)
    y = np.array(y)
    # Generate x data if it's empty
    if np.size(x) == 0:
        x = np.linspace(0, np.size(y), np.size(y))
        # 点が1つしかないとscatterがエラーをおこすため
    if np.size(x) == 1 and np.size(y) == 1:
        x = [x]
        y = [y]

    # marker style treatment
    if style[-1] == '.':  # 後ろがdotの場合
        if showfacecolor is False:  # dot指定があり，facecolorがoffの場合
            style = marksymdef[style[:-1]] + '-open-dot'
        else:  # dot指定があり,facecolorはon
            style = marksymdef[style[:-1]] + '-dot'
    else:  # 後ろがdotではない場合
        if showfacecolor is False:  # dot指定無しで，facecolorがoffの場合
            style = marksymdef[style] + '-open'
        else:  # dot指定なしで，facecolorはon
            style = marksymdef[style]

    # facecolor treatment
    temp = []
    if facecolor is not None:
        for i in range(len(facecolor)):
            if facecolor[i] in coldef:
                temp = np.append(temp, coldef[facecolor[i]])
            else:
                temp = np.append(temp, facecolor[i])
        facecolor = temp
    else:
        pass

    if facecolor is not None:
        if len(facecolor) == 1:
            temp = facecolor
            for i in range(len(x)):
                facecolor = np.append(facecolor, temp)
        else:
            pass

    # Colormap treatment when intensity is designated
    if colormap is not None:
        showscale = True

    if sizeref is not None:
        sizemode = 'area'

    if markersize is not None and size is None:
        size = markersize

    # Define marker propoeties
    marker = dict(symbol=style, color=facecolor, size=size, opacity=alpha,
                  showscale=showscale, colorscale=colormap, sizeref=sizeref, sizemode=sizemode,
                  line=dict(width=edgewidth, color=edgecolor))

    if figall[figcount-1]._has_subplots():  # if subplots exist
        if ax is None:  # no grid designation
            row = figrow
            col = figcol
        else:
            row = ax[0]
            col = ax[1]

    if figall[figcount-1]._has_subplots():  # if subplots exist
        figall[figcount-1].add_trace(go.Scattergl(x=x, y=y, mode='markers', marker=marker,
                                                    text=hover, name=dataname,
                                                    ), row=row, col=col
                                       )
    else:
        figall[figcount-1].add_trace(go.Scattergl(x=x, y=y, mode='markers', marker=marker,
                                                    text=hover, name=dataname,
                                                    )
                                       )

    # Update figure
    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% Bar function
def bar(x, y, ax=None, dataname=None, alpha=None, edgecolor=None, edgewidth=None, color=None,
        hover=None, label=None, textposition='auto', width=None, base=None, reverse=False, ori=None):
    global winall, figall, figcount, figcol, figrow
    if not any(figall):  # if there are no figure frame
        figure()

    if type(x) is not list:
        x = [x]
    if type(y) is not list:
        y = [y]

    if figall[figcount-1]._has_subplots():  # if subplots exist
        if ax is None:  # no grid designation
            row = figrow
            col = figcol
        else:
            row = ax[0]
            col = ax[1]

    if edgecolor is not None:
        if edgecolor in coldef:
            edgecolor = coldef[edgecolor]
        else:
            pass

    if color is not None:
        if type(color) is list or type(color) == np.ndarray:
            for i in range(np.size(color)):
                if color[i] in coldef:
                    color[i] = coldef[color[i]]
        else:
            if color in coldef:
                color = coldef[color]
            else:
                pass

    marker = dict(color=color, line_color=edgecolor, line_width=edgewidth)

    if reverse is True:
        y = np.array(y)
        base = [0] * np.size(y)
        for i in range(np.size(y)):
            base[i] = -y[i]

    if ori == 'h':
        xtemp = x
        x = y
        y = xtemp

    if figall[figcount-1]._has_subplots():  # if subplots exist
        figall[figcount-1].add_trace(go.Bar(x=x, y=y, name=dataname, hovertext=hover, opacity=alpha, marker=marker,
                                              text=label, width=width, base=base, orientation=ori), row=row, col=col
                                       )
    else:
        figall[figcount-1].add_trace(go.Bar(x=x, y=y, name=dataname, hovertext=hover, opacity=alpha, marker=marker,
                                              text=label, width=width, base=base, orientation=ori)
                                       )
        # figall[figcount-1].update_traces(marker=marker,opacity=alpha)

    # Update figure
    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% quivar function
import plotly.figure_factory as ff


def quiver(x, y, u, v, ax=None, color=None, scale=0.1, arrowscale=0.3, linewidth=None, dataname=None):
    global winall, figall, figcount, figcol, figrow  # figcol, figrow -> なくても参照可能？
    if not any(figall):  # if there are no figure frame (create_trisurfは実はfigure frameいらない)
        figure()
        return

    # 無くても問題なし
    """
    x = np.array(x)
    y = np.array(y)
    u = np.array(u)
    v = np.array(v)
    x = x.flatten()
    y = y.flatten()
    u = u.flatten()
    v = v.flatten()
    """

    if color is not None:
        if color in coldef:
            color = coldef[color]
        else:
            pass

    lineprop = dict(width=linewidth, color=color)

    figadd = ff.create_quiver(x, y, u, v, scale=scale, arrow_scale=arrowscale, line=lineprop, name=dataname)

    if figall[figcount-1]._has_subplots():  # if subplots exist
        if ax is None:  # no grid designation
            row = figrow
            col = figcol
        else:
            row = ax[0]
            col = ax[1]

    # aspect=dict(x=1, y=1, z=1)

    if figall[figcount-1]._has_subplots():  # if subplots exist
        figall[figcount-1].add_traces(data=figadd.data, row=row, col=col)
    else:
        figall[figcount-1].add_traces(data=figadd.data)
        # figall[figcount-1].update_layout(scene_aspectratio=aspect)
        # figall[figcount-1].update_layout(scene_aspectmode='auto')

    # Update figure
    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% streamline function
# import plotly.figure_factory as ff
def streamline(x, y, u, v, ax=None, color=None, density=1, arrowscale=.09,
               linewidth=None, dataname=None):
    global winall, figall, figcount, figcol, figrow  # figcol, figrow -> なくても参照可能？
    if not any(figall):  # if there are no figure frame (create_trisurfは実はfigure frameいらない)
        figure()
        return

    # 無くても問題なし
    """
    x = np.array(x)
    y = np.array(y)
    u = np.array(u)
    v = np.array(v)
    x = x.flatten()
    y = y.flatten()
    u = u.flatten()
    v = v.flatten()
    """

    if color is not None:
        if color in coldef:
            color = coldef[color]
        else:
            pass

    lineprop = dict(width=linewidth, color=color)

    figadd = ff.create_streamline(x, y, u, v, density=density, arrow_scale=arrowscale, line=lineprop, name=dataname)

    if figall[figcount-1]._has_subplots():  # if subplots exist
        if ax is None:  # no grid designation
            row = figrow
            col = figcol
        else:
            row = ax[0]
            col = ax[1]

    # aspect=dict(x=1, y=1, z=1)

    if figall[figcount-1]._has_subplots():  # if subplots exist
        figall[figcount-1].add_traces(data=figadd.data, row=row, col=col)
    else:
        figall[figcount-1].add_traces(data=figadd.data)
        # figall[figcount-1].update_layout(scene_aspectratio=aspect)
        # figall[figcount-1].update_layout(scene_aspectmode='auto')

    # Update figure
    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% 3D plotting function 
# %% Scatter3 function (plot3 function includes scatter3 function)
def scatter3(x, y, z, c=None, ax=None, facecolor=None, style='o', size=None, alpha=None, hover=None,
             sizemode=None, edgecolor=None, edgewidth=None, colormap=None, color=None,
             showscale=False, sizeref=None, dataname=None, showfacecolor=True):
    global winall, figall, figcount, figcol, figrow, subplotidx
    if not any(figall):  # if there are no figure frame
        figure()

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    # 点が1つしかないとscatterがエラーをおこす
    if np.size(x) == 1 and np.size(y) == 1 and np.size(z) == 1:
        x = [x]
        y = [y]
        z = [z]

    # marker style treatment
    if style[-1] == '.':  # 後ろがdotの場合
        if showfacecolor is False:  # dot指定があり，facecolorがoffの場合
            style = marksymdef[style[:-1]] + '-open-dot'
        else:  # dot指定があり,facecolorはon
            style = marksymdef[style[:-1]] + '-dot'
    else:  # 後ろがdotではない場合
        if showfacecolor is False:  # dot指定無しで，facecolorがoffの場合
            style = marksymdef[style] + '-open'
        else:  # dot指定なしで，facecolorはon
            style = marksymdef[style]

    # facecolor treatmen
    if type(facecolor) is list or type(facecolor) == np.ndarray:
        for i in range(len(facecolor)):
            if facecolor[i] in coldef:
                facecolor[i] = coldef[facecolor[i]]
            else:
                pass
    else:
        if facecolor in coldef:
            facecolor = coldef[facecolor]
        else:
            pass
    if type(color) is list or type(color) == np.ndarray:
        for i in range(len(color)):
            if color[i] in coldef:
                color[i] = coldef[color[i]]
            else:
                pass
    else:
        if color in coldef:
            color = coldef[color]
        else:
            pass
    if color is not None and facecolor is None:
        facecolor = color

    # Colormap treatment when intensity is designated
    if c is not None:
        facecolor = c
        showscale = True

    if sizeref is not None:
        sizemode = 'diameter'

    # Define marker propoeties
    marker = dict(symbol=style, color=facecolor, size=size, opacity=alpha,
                  showscale=showscale, colorscale=colormap, sizeref=sizeref, sizemode=sizemode,
                  line=dict(width=edgewidth, color=edgecolor))

    if figall[figcount-1]._has_subplots():  # if subplots exist
        idx = subplotidx
        if ax is None:  # no grid designation
            row = figrow
            col = figcol
        else:
            row = ax[0]
            col = ax[1]
    else:
        idx = 1

    if figall[figcount-1]._has_subplots():  # if subplots exist
        figall[figcount-1].add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=marker,
                                                    text=hover, name=dataname,
                                                    ), row=row, col=col
                                       )
    else:
        figall[figcount-1].add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=marker,
                                                    text=hover, name=dataname,
                                                    )
                                       )

    # control view
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=-0.5),
        eye=dict(x=1.35, y=-1.35, z=1.85)
    )  

    idx = 'scene' + str(idx)
    figall[figcount-1].layout[idx].camera = camera

    # Update figure
    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% 3D plot function
def plot3(x, y, z, ax=None, color=None, style=None, linewidth=None, dataname=None,
          connectgap=False, hover=None, hoverinfo=None, showfacecolor=True,
          markersize=None, markerfacecolor=None, markeredgecolor=None, markeredgewidth=None):

    global winall, figall, figcount, figcol, figrow
    if not any(figall):  # if there are no figure frame
        figure()

    # 点が1つしかないとscatterがエラーをおこすため，応急的な処置->良い方法ない？
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    # x = x.flatten()
    # y = y.flatten()
    # z = z.flatten()
    
    if figall[figcount-1]._has_subplots():  # if subplots exist
        idx = subplotidx
        if ax is None:  # no grid designation
            row = figrow
            col = figcol
        else:
            row = ax[0]
            col = ax[1]
    else:
        idx = 1    

    # when only one data is input (mode->'marker')
    if np.size(x) == 1 and np.size(y) == 1 and np.size(z) == 1:
        if style is None:
            style = 'o'
        # 点が1つしかないとscatterがエラーをおこす
        x = [x]
        y = [y]
        y = [z]

    if style is None:
        style = '-'

    # Detect line/marker style
    if style[0] in linestyledef:  # exist line style indicator
        mode = 'lines+markers'
        if style[0] == style[:]:  # '-' or ':'
            linestyle = linestyledef[style[0]]
            markerstyle = style.replace(style[0], '')
        else:  # other indicator exist
            if style[0:2] in linestyledef:  # exist second line style indicator
                linestyle = linestyledef[style[0:2]]
                markerstyle = style.replace(style[0:2], '')
            else:
                linestyle = linestyledef[style[0]]
                markerstyle = style.replace(style[0], '')
    else:  # not exist line-style indicator
        mode = 'markers'
        linestyle = None
        markerstyle = style
    # print(linestyle)

    if len(markerstyle) == 0:
        mode = 'lines'
        markerstyle = None

    if markerstyle is not None:
        if markerstyle[-1] == '.':  # exist dot indicator
            if showfacecolor is False:  # when markerfacecolor is off
                markerfacecolor = None  # 後の処理の為にNoneにしておく
                markerstyle = marksymdef[markerstyle[:-1]] + '-open-dot'
            else:  # when markerfacecolor is on
                markerstyle = marksymdef[markerstyle[:-1]] + '-dot'
        else:  # not exist dot indicator
            if showfacecolor == False:  # when markerfacecolor is off
                markerfacecolor = None  # 後の処理の為にNoneにしておく
                markerstyle = marksymdef[markerstyle] + '-open'
            else:  # when markerfacecolor is on
                markerstyle = marksymdef[markerstyle]

    # Detect line color
    if color is not None:
        if color in coldef:
            color = coldef[color]
        else:
            pass

    # Detect marker-face color
    if markerfacecolor is None:
        markerfacecolor = color
    else:
        if markerfacecolor in coldef:
            markerfacecolor = coldef[markerfacecolor]
        else:
            pass

    # Detect marker-edge color
    if markeredgecolor is None:
        markeredgecolor = color
    else:
        if markeredgecolor in coldef:
            markeredgecolor = coldef[markeredgecolor]
        else:
            pass

            # hover control
    if hover is not None:
        # hoverinfo = 'text+name'
        pass

    # Control line/marker properties
    if mode == 'lines':
        line = dict(color=color, width=linewidth, dash=linestyle)
    elif mode == 'markers':
        marker = dict(symbol=markerstyle, color=markerfacecolor, size=markersize,
                      line=dict(width=markeredgewidth, color=markeredgecolor))
    else:
        line = dict(color=color, width=linewidth, dash=linestyle)
        marker = dict(symbol=markerstyle, color=markerfacecolor, size=markersize,
                      line=dict(width=markeredgewidth, color=markeredgecolor))

    if figall[figcount-1]._has_subplots():  # if subplots exist
        if ax is None:  # no grid designation
            row = figrow
            col = figcol
        else:
            row = ax[0]
            col = ax[1]

    if figall[figcount-1]._has_subplots():  # if subplots exist
        if mode == 'lines':
            figall[figcount-1].add_trace(go.Scatter3d(x=x, y=y, z=z, mode=mode, line=line, name=dataname
                                                        ), row=row, col=col)
        elif mode == 'markers':
            figall[figcount-1].add_trace(go.Scatter3d(x=x, y=y, z=z, mode=mode, marker=marker, name=dataname),
                                           row=row, col=col)
        else:
            figall[figcount-1].add_trace(go.Scatter(x=x, y=y, z=z, mode=mode, line=line, marker=marker,
                                                      name=dataname
                                                      ), row=row, col=col
                                           )
    else:
        if mode == 'lines':
            figall[figcount-1].add_trace(go.Scatter3d(x=x, y=y, z=z, mode=mode, line=line, name=dataname,
                                                        connectgaps=connectgap, )
                                           )
        elif mode == 'markers':
            figall[figcount-1].add_trace(go.Scatter3d(x=x, y=y, z=z, mode=mode, marker=marker, name=dataname,
                                                        )
                                           )
        else:
            figall[figcount-1].add_trace(go.Scatter3d(x=x, y=y, z=z, mode=mode, line=line, marker=marker, name=dataname,
                                                        connectgaps=connectgap, text=hover, hoverinfo=hoverinfo),
                                           )

    # control view
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=-0.5),
        eye=dict(x=1.35, y=-1.35, z=1.85)
    )  

    idx = 'scene' + str(idx)
    figall[figcount-1].layout[idx].camera = camera
    
    # Update figure
    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% Surf
def surf(x, y, z, facecolor=None, showscale=False, showcontour=False, contourcolor=None, barxposition=None,
         highlightcolor=None, zproject=False, ax=None, levels=[None, None], colormap=None):
    global winall, figall, figcount, figcol, figrow, subplotidx

    if not any(figall):  # if there are no figure frame
        print('No figure')
        return

    z = np.array(z)
    if len(x) == 0:
        x = None
    if len(y) == 0:
        y = None

    # control contourcolor
    if contourcolor is None:
        usecolormap = True
    else:
        contourcolor = coldef[contourcolor]
        usecolormap = False
    if highlightcolor is None:
        pass
    else:
        highlightcolor = coldef[highlightcolor]

    if figall[figcount-1]._has_subplots():  # if subplots exist
        idx = subplotidx
        if ax is None:  # no grid designation
            row = figrow
            col = figcol
        else:
            row = ax[0]
            col = ax[1]
    else:
        idx = 1

    if figall[figcount-1]._has_subplots():  # if subplots exist
        figall[figcount-1].add_trace(
            go.Surface(z=z, x=x, y=y, showscale=showscale, surfacecolor=facecolor, colorbar_x=barxposition, colorscale=colormap,
                       contours={
                           "z": {"show": showcontour, "start": levels[0], "end": levels[1], "size": 0.05, "color": contourcolor}
                       }
                       ), row=row, col=col
        )

        fig.update_traces(contours_z=dict(show=showcontour, usecolormap=usecolormap,
                                          highlightcolor=highlightcolor, project_z=zproject), row=row, col=col)

    else:  # No subplot
        if len(figall[figcount-1].data) == 0:  # No data
            figall[figcount-1].add_trace(
                go.Surface(z=z, x=x, y=y, showscale=showscale, surfacecolor=facecolor, colorbar_x=barxposition, colorscale=colormap,
                           contours={
                               "z": {"show": showcontour, "start": levels[0], "end": levels[1], "size": 0.05, "color": contourcolor}
                           }
                           )
            )
        else:  # data exist
            figall[figcount-1].add_traces(
                data=[go.Surface(z=z, x=x, y=y, showscale=showscale, surfacecolor=facecolor, colorbar_x=barxposition, colorscale=colormap,
                                 contours={
                                     "z": {"show": showcontour, "start": levels[0], "end": levels[1], "size": 0.05, "color": contourcolor}
                                 }
                                 )
                      ]
            )
        fig.update_traces(contours_z=dict(show=showcontour, usecolormap=usecolormap,
                                          highlightcolor=highlightcolor, project_z=zproject))

    # control view
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=-0.5),
        eye=dict(x=1.35, y=-1.35, z=1.85)
    )  

    idx = 'scene' + str(idx)
    figall[figcount-1].layout[idx].camera = camera

    # Update figure
    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% mesh function
def mesh(vertices, triangles=None, intensity=None, showscale=False, intensitymode=None,
         ax=None, alpha=None, facecolor=None, hypermesh=-1, colormap=None, showgridedge=True):
    global winall, figall, figcount, figcol, figrow, subplotidx  # figcol, figrow -> なくても参照可能？

    if not any(figall):  # if there are no figure frame
        print('No figure')
        return

    if figall[figcount-1]._has_subplots():  # if subplots exist
        idx = subplotidx
        if ax is None:  # no grid designation
            row = figrow
            col = figcol
        else:
            row = ax[0]
            col = ax[1]
    else:
        idx = 1

    vertices = np.array(vertices)
    if vertices.shape[0] < vertices.shape[1]:
        vertices = vertices.T
    x = np.array(vertices[:, 0])
    y = np.array(vertices[:, 1])
    z = np.array(vertices[:, 2])

    if triangles is None:
        i = None
        j = None
        k = None
    else:
        triangles = np.array(triangles)
        if triangles.shape[0] < triangles.shape[1]:
            triangles = triangles.T
        i = np.array(triangles[:, 0])
        j = np.array(triangles[:, 1])
        k = np.array(triangles[:, 2])

    if facecolor is not None:
        facecolor = coldef[facecolor]

    # colormapが指定されるとcolormapの方が優先される
    # colormapが指定された場合，defaultでshowscaleとなる

    if intensity is not None:
        if intensitymode is None:
            if len(intensity) == vertices.shape[0]:
                intensitymode = 'vertex'
            elif len(intensity) == triangles.shape[0]:
                intensitymode = 'cell'

    # plot surface triangulation
    if triangles is not None:
        tri_vertices = vertices[triangles]
        Xe = []
        Ye = []
        Ze = []
        for T in tri_vertices:
            Xe += [T[k % 3][0] for k in range(4)] + [None]
            Ye += [T[k % 3][1] for k in range(4)] + [None]
            Ze += [T[k % 3][2] for k in range(4)] + [None]
    else:
        Xe = None
        Ye = None
        Ze = None

    if figall[figcount-1]._has_subplots():  # if subplots exist
        figall[figcount-1].add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=facecolor, intensity=intensity, intensitymode=intensitymode,
                                                 opacity=alpha, colorscale=colormap, alphahull=hypermesh, showscale=showscale,
                                                 ), row=row, col=col
                                       )
        if showgridedge == True:
            plot3(Xe, Ye, Ze, color='rgb(40,40,40)', linewidth=0.5)

    else:  # No subplot
        if len(figall[figcount-1].data) == 0:  # No data
            figall[figcount-1].add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=facecolor, intensity=intensity, intensitymode=intensitymode,
                                                     opacity=alpha, colorscale=colormap, alphahull=hypermesh, showscale=showscale,
                                                     )
                                           )

        else:  # data exist
            figall[figcount-1].add_traces(
                data=[go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=facecolor, intensity=intensity, intensitymode=intensitymode,
                                opacity=alpha, colorscale=colormap, alphahull=hypermesh, showscale=showscale,
                                )
                      ]
            )
        if showgridedge == True:
            plot3(Xe, Ye, Ze, color='rgb(40,40,40)', linewidth=0.5)

    # control view
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=-0.5),
        eye=dict(x=1.35, y=-1.35, z=1.85)
    )  
    
    idx = 'scene' + str(idx)
    figall[figcount-1].layout[idx].camera = camera

    # Update figure
    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


def colorbar(title=None, ax=None):
    global winall, figall, figcount, figcol, figrow, subplotidx  # figcol, figrow -> なくても参照可能？

    if not any(figall):  # if there are no figure frame
        print('No figure')
        return

    if figall[figcount-1]._has_subplots():  # if subplots exist
        idx = subplotidx
        if ax is None:  # no grid designation
            row = figrow
            col = figcol
        else:
            row = ax[0]
            col = ax[1]
    else:
        idx = 1

    for data in figall[figcount-1].data:
        try:
            data.showscale = True
            data.colorbar.title.text = title
        except:
            pass

    # Update figure
    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% Trisurf function
import matplotlib.tri as mtri
# import plotly.figure_factory as ff
from scipy.spatial import Delaunay


def trisurf(vertices, simplices=None, c=None, ax=None, colormap=None, showscale=True,
            showgridedge=True):
    global winall, figall, figcount, figcol, figrow  # figcol, figrow -> なくても参照可能？
    if not any(figall):  # if there are no figure frame (create_trisurfは実はfigure frameいらない)
        print('No figure')
        return

    # Generate simplices?
    if simplices is None:
        simplices = mtri.Triangulation(vertices[0], vertices[1]).triangles
    else:
        simplices = np.array(simplices)
        if simplices.shape[0] < simplices.shape[1]:
            simplices = simplices.T

        if simplices.shape[1] == 2:
            u = np.array(simplices[:, 0])
            v = np.array(simplices[:, 1])
            points2D = np.vstack([u, v]).T
            tri = Delaunay(points2D)
            simplices = tri.simplices

    vertices = np.array(vertices)
    if vertices.shape[0] < vertices.shape[1]:
        vertices = vertices.T
    x = np.array(vertices[:, 0])
    y = np.array(vertices[:, 1])
    z = np.array(vertices[:, 2])

    if colormap is not None:
        if type(colormap) is not list:
            if colormap in colmapdef:
                colormap = colmapdef[colormap]
            else:
                pass
        else:
            pass

    figadd = ff.create_trisurf(x=x, y=y, z=z, simplices=simplices, plot_edges=showgridedge,
                               colormap=colormap, show_colorbar=showscale, color_func=c)

    if figall[figcount-1]._has_subplots():  # if subplots exist
        idx = subplotidx
        if ax is None:  # no grid designation
            row = figrow
            col = figcol
        else:
            row = ax[0]
            col = ax[1]
    else:
        idx = 1

    # aspect=dict(x=1, y=1, z=1)

    if figall[figcount-1]._has_subplots():  # if subplots exist
        figall[figcount-1].add_traces(data=figadd.data, row=row, col=col)
    else:
        figall[figcount-1].add_traces(data=figadd.data)
        # figall[figcount-1].update_layout(scene_aspectratio=aspect)
        # figall[figcount-1].update_layout(scene_aspectmode='auto')

    # control view
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=-0.5),
        eye=dict(x=1.35, y=-1.35, z=1.85)
    )  

    idx = 'scene' + str(idx)
    figall[figcount-1].layout[idx].camera = camera

    # Update figure
    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% 3D Cone function
# anchor="tip"
def cone(x, y, z, u, v, w, ax=None, scale=1, size=None, sizemode="absolute", normalized=False,
         dataname=None, alpha=None, showscale=True, lightoption=None, colormap=None):
    global winall, figall, figcount, figcol, figrow  # figcol, figrow -> なくても参照可能？
    if not any(figall):  # if there are no figure frame
        figure()

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    u = np.array(u)
    v = np.array(v)
    w = np.array(w)

    if normalized is True:
        vecnorm = np.sqrt(u ** 2 + v ** 2 + w ** 2)
        u = scale * u / vecnorm
        v = scale * v / vecnorm
        w = scale * w / vecnorm
    if figall[figcount-1]._has_subplots():  # if subplots exist
        idx = subplotidx
        if ax is None:  # no grid designation
            row = figrow
            col = figcol
        else:
            row = ax[0]
            col = ax[1]
    else:
        idx = 1

    if figall[figcount-1]._has_subplots():  # if subplots exist
        figall[figcount-1].add_trace(go.Cone(x=x, y=y, z=z, u=u, v=v, w=w, sizemode=sizemode, sizeref=size, lighting=lightoption,
                                               name=dataname, opacity=alpha, showscale=showscale, colorscale=colormap), row=row, col=col
                                       )

    else:  # No subplot
        if len(figall[figcount-1].data) == 0:  # No data
            figall[figcount-1].add_trace(go.Cone(x=x, y=y, z=z, u=u, v=v, w=w, sizemode=sizemode, sizeref=size, lighting=lightoption,
                                                   name=dataname, opacity=alpha, showscale=showscale, colorscale=colormap)
                                           )
        else:  # data exist
            # print('Hello')
            figall[figcount-1].add_traces(data=[go.Cone(x=x, y=y, z=z, u=u, v=v, w=w, sizemode=sizemode, sizeref=size, lighting=lightoption,
                                                          name=dataname, opacity=alpha, showscale=showscale, colorscale=colormap)
                                                  ]
                                            )

    # control view
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=-0.5),
        eye=dict(x=1.35, y=-1.35, z=1.85)
    )  

    idx = 'scene' + str(idx)
    figall[figcount-1].layout[idx].camera = camera

    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% 3D isosurface function
def isosurf(x, y, z, v, ax=None, isorange=None, xcap=True, ycap=True, xycap=True, surfcount=None,
            alpha=None, fill=None, zslice=None, yslice=None, fillpattern=None, colormap=None, showscale=None):
    global winall, figall, figcount, figcol, figrow, subplotidx  # figcol, figrow -> なくても参照可能？

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    v = np.array(v)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    v = v.flatten()

    if not any(figall):  # if there are no figure frame
        figure()

    if figall[figcount-1]._has_subplots():  # if subplots exist
        idx = subplotidx
        if ax is None:  # no grid designation
            row = figrow
            col = figcol
        else:
            row = ax[0]
            col = ax[1]
    else:
        idx = 1

    if isorange is not None:
        isomin = isorange[0]
        isomax = isorange[1]
    else:
        isomin = None
        isomax = None

    if xycap is False:
        caps = dict(x_show=False, y_show=False)
    else:
        caps = dict(x_show=xcap, y_show=ycap)

    surface = dict(count=surfcount, fill=fill, pattern=fillpattern)

    if zslice is not None:
        slices_z = dict(show=True, locations=zslice)
    else:
        slices_z = dict(show=False)

    if yslice is not None:
        slices_y = dict(show=True, locations=yslice)
    else:
        slices_y = dict(show=False)

    # surf_count -> number of isosurfaces, 2 by default: only min and max
    if figall[figcount-1]._has_subplots():  # if subplots exist
        figall[figcount-1].add_trace(go.Isosurface(x=x, y=y, z=z, value=v, isomin=isomin, isomax=isomax, opacity=alpha,
                                                     caps=caps, surface=surface, slices_z=slices_z, slices_y=slices_y,
                                                     colorscale=colormap, showscale=showscale), row=row, col=col
                                       )

    else:  # No subplot
        if len(figall[figcount-1].data) == 0:  # No data
            figall[figcount-1].add_trace(go.Isosurface(x=x, y=y, z=z, value=v, isomin=isomin, isomax=isomax, opacity=alpha,
                                                         caps=caps, surface=surface, slices_z=slices_z, slices_y=slices_y,
                                                         colorscale=colormap, showscale=showscale)
                                           )
        else:  # data exist
            figall[figcount-1].add_traces(data=[go.Isosurf(x=x, y=y, z=z, value=v, isomin=isomin, isomax=isomax, opacity=alpha,
                                                             caps=caps, surface=surface, slices_z=slices_z, slices_y=slices_y,
                                                             colorscale=colormap, showscale=showscale)
                                                  ]
                                            )

    # control view
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=-0.5),
        eye=dict(x=1.35, y=-1.35, z=1.85)
    )  

    idx = 'scene' + str(idx)
    figall[figcount-1].layout[idx].camera = camera

    # Update figure
    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


# %% quiver3 function 
def quiver3(x, y, z, xdir, ydir, zdir, ax=None, showmarker=False, markerstyle='o', color='b', linestyle='-',
            linewidth=None, markersize=4, scale=1, conesize=0.2, markercolor=None):

    global winall, figall, figcount, figcol, figrow  # figcol, figrow -> なくても参照可能？
    if not any(figall):  # if there are no figure frame
        figure()

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    xdir = np.array(xdir)
    ydir = np.array(ydir)
    zdir = np.array(zdir)
    
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    xdir = xdir.flatten()
    ydir = ydir.flatten()
    zdir = zdir.flatten()

    if color is not None:
        if color in coldef:
            color = coldef[color]
        else:
            pass
        
    usermap = [[0, color], [0.5, color], [1, color]]
    if markercolor is None:
        markercolor = color
    else:
        if markercolor in coldef:
            markercolor = coldef[markercolor]
        else:
            pass
        
    line = dict(color=color, width=linewidth, dash='solid')        

    for i in range(np.size(x)):
        figall[figcount-1].add_trace(go.Scatter3d(x=[x[i],x[i]+xdir[i]], 
                                                  y=[y[i],y[i]+ydir[i]], 
                                                  z=[z[i],z[i]+zdir[i]], mode='lines', line=line))
        
    figall[figcount-1].add_trace(go.Cone(x=x+xdir, y=y+ydir, z=z+zdir, u=xdir, v=ydir, w=zdir, showscale=False))
    
    if showmarker is False:
         pass
    else:
         scatter3(x, y, z, ax=ax, facecolor=markercolor, style=markerstyle, size=markersize)

    if showPlot[figcount-1]: winall[figcount-1].update(figall[figcount-1])


def showfig(config=None, exec=False):
    win = PlotlyViewer(fig, config=config, exec=exec)
    winall.append(win)

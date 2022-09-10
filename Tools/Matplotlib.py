
__author__ = "Anthony Beaucamp"
__copyright__ = "Copyright (C) 2020 Kyoto University"
__license__ = "Public Domain"
__version__ = "1.0"

# "Matlab" style functions for plotting


import os, sys
import ctypes
from numpy import *
import atexit
import builtins
from threading import Thread
from operator import itemgetter
import subprocess, time

import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.interactive(True)
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colorbar as colorbar
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.axes import Axes

ImageWidth = 24
ImageHeight = 24

global figNumber
figNumber = None

from base64 import b64decode, b64encode
# from zlib import decompress, compress
from io import BytesIO
from OCC.Tools.PlottingEmbededImages import *

cwd = os.path.dirname(os.path.realpath(__file__))

# def EncodeImage(cwd, fileName):
#i
#     with open(os.path.join(cwd, fileName), "rb") as fileObj:
#         data = b64encode(fileObj.read())
#
#     return data

def DecodeImageToStream(encodedImage):
    imageData = b64decode(encodedImage)
    stream = BytesIO(bytearray(imageData))
    return stream

def BitmapFromImage(filename):
    """!
    Convert an image to a bitmap
    @param filename: image file path on disk
    """
    image = wx.Image(os.path.join(cwd, "%s" % filename), wx.BITMAP_TYPE_PNG)
    image = image.Scale(ImageWidth, ImageHeight, wx.IMAGE_QUALITY_HIGH)
    return image.ConvertToBitmap()

#####################################################################
# Massive hacks to achieve orthogonal projection and axes at min side
class ModifiedAxes3D(Axes3D):
    def __init__(self, baseObject):
        """!
        Default constructor
        @param baseObject: object
        """
        self.__class__ = type(baseObject.__class__.__name__, (self.__class__, baseObject.__class__), {})
        self.__dict__ = baseObject.__dict__
        self.mouse_init()

    def set_some_features_visibility(self, visible):
        """!
        Toggle the visibility of certain features
        @param visible: bool
        """
        for t in self.w_zaxis.get_ticklines() + self.w_zaxis.get_ticklabels():
            t.set_visible(visible)
        self.w_zaxis.line.set_visible(visible)
        self.w_zaxis.pane.set_visible(visible)
        self.w_zaxis.label.set_visible(visible)

    def draw(self, renderer):
        # draw the background patch
        self.patch.draw(renderer)
        # self.axesPatch.draw(renderer)
        self._frameon = False

        # first, set the aspect
        locator = self.get_axes_locator()
        if locator:
            pos = locator(self, renderer)
            self.apply_aspect(pos)
        else:
            self.apply_aspect()

        # add the projection matrix to the renderer
        self.M = self.get_proj()
        renderer.M = self.M
        renderer.vvec = self.vvec
        renderer.eye = self.eye
        renderer.get_axis_position = self.get_axis_position

        # Calculate projection of collections and zorder them
        zlist = [(col.do_3d_projection(renderer), col) \
                 for col in self.collections]
        zlist.sort(key=itemgetter(0), reverse=True)
        for i, (z, col) in enumerate(zlist):
            col.zorder = i

        # Calculate projection of patches and zorder them
        zlist = [(patch.do_3d_projection(renderer), patch) \
                 for patch in self.patches]
        zlist.sort(key=itemgetter(0), reverse=True)
        for i, (z, patch) in enumerate(zlist):
            patch.zorder = i

        # Sort axis location
        self.xaxis._axinfo['juggled'] = (0,0,0)
        self.yaxis._axinfo['juggled'] = (0,0,0)
        self.zaxis._axinfo['juggled'] = (2,2,2)
        
        # Sort grid planes
        if self._axis3don:
            axes = (self.xaxis, self.yaxis, self.zaxis)
            # Draw panes first
            # for ax in axes:
            #    ax.draw_pane(renderer)
            # Then axes
            for ax in axes:
                ax.draw(renderer)

        # Then rest
        Axes.draw(self, renderer)

        #        zaxis = self.zaxis
        #        draw_grid_old = zaxis.axes._draw_grid

        # disable draw grid
        #        zaxis.axes._draw_grid = False

        # draw zaxis on the left side
        #        tmp_planes = zaxis._PLANES
        #        zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
        #                         tmp_planes[0], tmp_planes[1],
        #                         tmp_planes[4], tmp_planes[5])
        #        zaxis.draw(renderer)
        #        zaxis._PLANES = tmp_planes

        # disable draw grid


# zaxis.axes._draw_grid = draw_grid_old


def inv_transform(xs, ys, zs, M):
    """!
    Apply an inverse transform
    @param xs: float xPosition
    @param ys: float yPosition
    @param zs: float zPosition
    @param M: Matrix
    """
    try:
        iM = linalg.inv(M)
        vec = vec_pad_ones(xs, ys, zs)
        vecr = np.dot(iM, vec)
        vecr = vecr / vecr[3]
    except:
        return 0, 0, 1
    return vecr[0], vecr[1], vecr[2]


def orthogonal_proj(zfront, zback):
    """!
    Create an orthogonal projection matrix
    @param zfront:
    @param zback:
    @return: matrix
    """
    a = (zfront + zback) / (zfront - zback)
    b = -2 * (zfront * zback) / (zfront - zback)
    return array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, a, b],
                  [0, 0, 0, zback]])

proj3d.inv_transform = inv_transform
proj3d.persp_transformation = orthogonal_proj


#####################################################################

def XYView(event):
    for ax in event.EventObject.Parent.canvas.figure.get_axes():
        ax.view_init(azim=-90, elev=90)
    plt.draw()

def XZView(event):
    for ax in event.EventObject.Parent.canvas.figure.get_axes():
        ax.view_init(azim=-90, elev=0)
    plt.draw()

def YZView(event):
    for ax in event.EventObject.Parent.canvas.figure.get_axes():
        ax.view_init(azim=0, elev=0)
    plt.draw()

def ResetView(event):
    for ax in event.EventObject.Parent.canvas.figure.get_axes():
        ax.view_init(azim=-45, elev=45)
    plt.draw()

def copyview(ax1, ax2):
    """!
    Copy axes limits and camera from ax1 to ax2
    @param ax1: axis handle
    @param ax2: axis handle
    @param PlottingEngine: int 0 = matplotlib, 1 = MATLAB (default)
    """

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_zlim(ax1.get_zlim())
    plt.draw()
    
def daspect(ax, x=1, y=1, z=1):
    """!
    Sets the data aspect ratio for the current axes
    @param ax: axis handle
    @param x:
    @param y:
    @param z:
    @param PlottingEngine: int 0 = matplotlib, 1 = MATLAB (default)
    """

    # Copy axes limits and camera
    try:
        aspects = [x, y, z]
        extents = array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        sz = extents[:, 1] - extents[:, 0]
        centers = mean(extents, axis=1)
        maxsize = max(abs(sz))
        for ctr, asp, dim in zip(centers, aspects, 'xyz'):
            getattr(ax, 'set_{}lim'.format(dim))(asp * (ctr - maxsize / 2), asp * (ctr + maxsize / 2))
    except:
        aspects = [x, y]
        extents = array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xy'])
        sz = extents[:, 1] - extents[:, 0]
        centers = mean(extents, axis=1)
        maxsize = max(abs(sz))
        for ctr, asp, dim in zip(centers, aspects, 'xy'):
            getattr(ax, 'set_{}lim'.format(dim))(asp * (ctr - maxsize / 2), asp * (ctr + maxsize / 2))    

def figureThread():
    # Refresh in thread
    plt.show()
    
def figure(size=[800,600], grid=[1,1], projection='2d', title=None, noThread=False, mixedPlots2D3D=False, visible=True, **kwargs):
    """!
    Create a figure which will contain the plots
    @param size: Size of the figure
    @param grid: list - Dimension of the number of plots (mxn subplots in the figure - default is [1,1])
    @param projection: str - '2d' or '3d'
    @param title: str - Title of the figure
    @param noThread: bool
    @param mixedPlots2D3D: bool
    @param PlottingEngine: int 0 = matplotlib, 1 = MATLAB(default)#
    @param kwargs:
    @return: return axes handle
    """

    # Create figure
    scale = float(ctypes.windll.shcore.GetScaleFactorForDevice(0)) / 100
    fig = plt.figure(figsize=(ceil(size[0]/80), ceil(size[1]/80)), dpi=80, facecolor='white', **kwargs)
    if title is not None:
        fig.canvas.set_window_title(title)

    try:
        # Edit toolbar
        tbar = fig.canvas.toolbar
        tbar.DeleteToolByPos(7)
        tbar.DeleteToolByPos(2)
        tbar.DeleteToolByPos(1)
        if projection is '3d':
            tbar.DeleteToolByPos(2)
            tbar.DeleteToolByPos(2)
            tbar.DeleteToolByPos(0)
            ON_XYView = wx.NewId()

            topViewStream   = DecodeImageToStream(topViewStr)
            rightViewStream = DecodeImageToStream(rightViewStr)
            rearViewStream  = DecodeImageToStream(rearViewStr)
            resetViewStream = DecodeImageToStream(resetViewStr)

            topViewImage = wx.ImageFromStream(topViewStream, wx.BITMAP_TYPE_ANY)
            topViewImage = topViewImage.Scale(ImageWidth, ImageHeight, wx.IMAGE_QUALITY_HIGH)
            topViewBmp   = wx.BitmapFromImage(topViewImage)

            rearviewImage = wx.ImageFromStream(rearViewStream, wx.BITMAP_TYPE_ANY)
            rearviewImage = rearviewImage.Scale(ImageWidth, ImageHeight, wx.IMAGE_QUALITY_HIGH)
            rearviewBmp   = wx.BitmapFromImage(rearviewImage)

            resetviewImage = wx.ImageFromStream(resetViewStream, wx.BITMAP_TYPE_ANY)
            resetviewImage = resetviewImage.Scale(ImageWidth, ImageHeight, wx.IMAGE_QUALITY_HIGH)
            resetviewBmp   = wx.BitmapFromImage(resetviewImage)

            rightviewImage = wx.ImageFromStream(rightViewStream, wx.BITMAP_TYPE_ANY)
            rightviewImage = rightviewImage.Scale(ImageWidth, ImageHeight, wx.IMAGE_QUALITY_HIGH)
            rightviewBmp   = wx.BitmapFromImage(rightviewImage)


            # tbar.AddSimpleTool(ON_XYView, BitmapFromImage('topview2.png'), 'XY View', '')
            # img = wx.Image(stream, wx.BITMAP_TYPE_ANY)
            # bitmap = wx.Bitmap(image)
            tbar.AddSimpleTool(ON_XYView, topViewBmp, 'XY View', '')
            wx.EVT_TOOL(tbar, ON_XYView, XYView)
            ON_XZView = wx.NewId()
            tbar.AddSimpleTool(ON_XZView, rightviewBmp, 'XZ View', '')
            # tbar.AddSimpleTool(ON_XZView, BitmapFromImage('rightview2.png'), 'XZ View', '')
            wx.EVT_TOOL(tbar, ON_XZView, XZView)
            ON_YZView = wx.NewId()
            tbar.AddSimpleTool(ON_YZView, rearviewBmp, 'YZ View', '')
            # tbar.AddSimpleTool(ON_YZView, BitmapFromImage('rearview2.png'), 'YZ View', '')
            wx.EVT_TOOL(tbar, ON_YZView, YZView)
            ON_ResetView = wx.NewId()
            tbar.AddSimpleTool(ON_ResetView, resetviewBmp, 'Reset View', '')
            # tbar.AddSimpleTool(ON_ResetView, BitmapFromImage('resetview2.png'), 'Reset View', '')
            wx.EVT_TOOL(tbar, ON_ResetView, ResetView)
            tbar.Realize()
    except:
        pass

    # Create axes
    ax = []
    for r in range(grid[0]):
        ax.append([])
        for c in range(grid[1]):
            if projection is '3d':
                mAx = fig.add_subplot(grid[0], grid[1], r * grid[1] + c + 1, projection='3d')
                mAx = fig.add_axes(ModifiedAxes3D(mAx))
                mAx.view_init(azim=-45, elev=45)
                ax[r].append(mAx)
            else:
                ax[r].append(fig.add_subplot(grid[0], grid[1], r*grid[1]+c+1))

    # Re-format and run in Thread
    if projection is '3d' and not mixedPlots2D3D:
        fig.tight_layout()

    if noThread:
        return ax, fig, plt

    thread = Thread(target=figureThread())
    thread.setDaemon(1)
    thread.start()

    if mixedPlots2D3D:
        return ax, fig
    return ax

def plot(ax, x, y=[], z=[], color='b', marker=None, **kwargs):
    """!
    Makes a 1D/2D/3D line plot from the x, y, z coordinates
    @param ax: axis handle
    @param x: list of x-coordinate values
    @param y: list of y-coordinate values
    @param z: list of z-coordinate values
    @param color: color of the line
    @param marker: marker for the vertices given by (x,y,z)
    @param PlottingEngine: int 0 = matplotlib, 1 = MATLAB(default)
    @param kwargs:
    @return: return axis handle
    """
    
    # Auto-generate Figure?
    if not any(ax):
        if not any(z):
            ax = Figure(size=[800, 600], grid=[1, 1], projection='2d')
        else:
            ax = Figure(size=[800, 600], grid=[1, 1], projection='3d')
        ax = ax[0][0]

    # Detect specified Vectors
    if not any(y):
        handle = ax.plot(x, c=color, marker=marker, label='curve', **kwargs)
    elif not any(z):
        handle = ax.plot(x, y, c=color, marker=marker, label='curve', **kwargs)
    else:
        handle = ax.plot(x, y, z, c=color, marker=marker, label='curve', **kwargs)
    plt.draw()
    return handle
    
def surf(ax, x, y, z, facecolors=None, alpha=1.0, cmap=plt.cm.jet, zLabel = '', **kwargs):
    """!
    Makes a three-dimensional surface plot.
    @param ax: axis handle
    @param x: list of x-coordinate values
    @param y: list of y-coordinate values
    @param z: list of z-coordinate values
    @param facecolors: list of facecolors
    @param alpha: float - transparency (between 0 and 1. default is 1.0)
    @param cmap: Colormap
    @param PlottingEngine: int 0 = matplotlib, 1 = MATLAB(default)
    @param kwargs:
    @return: return axis handle
    """
    
    if facecolors is not None:
        mask = ~isnan(facecolors)
        colors = (facecolors - amin(facecolors[mask])) / (amax(facecolors[mask]) - amin(facecolors[mask]))
    else:
        mask = ~isnan(z)
        colors = (z - amin(z[mask])) / (amax(z[mask]) - amin(z[mask]))

    # Auto-generate Figure?
    if ax is None:
        if z is None:
            ax = Figure(size=[800, 600], grid=[1, 1], projection='2d')
        else:
            ax = Figure(size=[800, 600], grid=[1, 1], projection='3d')
        ax = ax[0][0]

    # Produce plot of transposed data (to draw from back to front)
    handle = ax.plot_surface(transpose(x), transpose(y), transpose(z), facecolors=cmap(transpose(colors)), alpha=alpha, linewidth=0.0, antialiased=True, **kwargs)
    plt.draw()
    return handle

def scatter(ax, x, y=[], z=[], color='b', marker='o', area=20, **kwargs):
    """!
    Displays markers (circles) at the locations specified by the vectors x, y, and z.
    @param ax: axis handle
    @param x: list of x-coordinate values
    @param y: list of y-coordinate values
    @param z: list of z-coordinate values
    @param color: str - color of the marker ('r'-red, 'g'-green etc.)
    @param marker: marker symbol ('o' - circle as default)
    @param area: float - area of each marker
    @param PlottingEngine: int 0 = matplotlib, 1 = MATLAB(default)
    @param kwargs:
    @return: return axis handle
    """
    
    # Auto-generate Figure?
    if not any(ax):
        if not any(z):
            ax = Figure(size=[800, 600], grid=[1, 1], projection='2d')
        else:
            ax = Figure(size=[800, 600], grid=[1, 1], projection='3d')
        ax = ax[0][0]

    # Detect specified Vectors
    if not any(y):
        handle = ax.scatter(mgrid[0:len(x):1], x, s=area, c=color, marker=marker, **kwargs)
    elif not any(z):
        handle = ax.scatter(x, y, s=area, c=color, marker=marker, **kwargs)
    else:
        handle = ax.scatter(x, y, z, s=area, c=color, marker=marker, **kwargs)
    plt.draw()
    return handle

def quiver(ax, x, y, z, u, v, w, color='b', length=1.0, linewidth=1.0):
    """!
    Plots vectors with directions determined by components (u,v,w) at points determined by (x,y,z)
    @param ax: axis handle
    @param x: list of x-coordinate values
    @param y: list of y-coordinate values
    @param z: list of z-coordinate values
    @param u: list of x-value of vector
    @param v: list of y-value of vector
    @param w: list of z-value of vector
    @param color: str - color of the vector ('r'-red, 'g'-green etc.)
    @param length: Length of the vectors
    @param linewidth: float - Line width
    @param PlottingEngine: int 0 = matplotlib, 1 = MATLAB(default)
    @return: return axis handle
    """
    
    # Auto-generate Figure?
    if ax is None:
        if z is None:
            ax = Figure(size=[800,600], grid=[1,1], projection='2d')
        else:
            ax = Figure(size=[800,600], grid=[1,1], projection='3d')
        ax = ax[0][0]

    # Draw vectors
    ax.quiver(x+u*length, y+v*length, z+w*length, u, v, w, color=color, length=length, linewidth=1.0)
    plt.draw()

def quiver2d(ax, x, y, u, v, length=1.0, color='b', **kwargs):
    """!
    Plots 2d vectors with directions determined by components (u,v) at points determined by (x,y)
    @param ax: axis handle
    @param x: list of x-coordinate values
    @param y: list of y-coordinate values
    @param u: list of x-value of vector
    @param v: list of y-value of vector
    @param length: Length of the vectors
    @param color: str - color of the vector ('r'-red, 'g'-green etc.)
    @param PlottingEngine: int 0 = matplotlib, 1 = MATLAB(default)
    @param kwargs:
    @return: return axis handle
    """
    
    # Draw vectors
    handle = ax.quiver(x, y, u, v, color=color, **kwargs)
    plt.draw()
    return handle

def quiver3d(ax, x, y, z, u, v, w, length=1.0, color='b', **kwargs):
    """!
    Plots vectors with directions determined by components (u,v,w) at points determined by (x,y,z)
    @param ax: axis handle
    @param x: list of x-coordinate values
    @param y: list of y-coordinate values
    @param z: list of z-coordinate values
    @param u: list of x-value of vector
    @param v: list of y-value of vector
    @param w: list of z-value of vector
    @param length: Length of the vectors
    @param color: str - color of the vector ('r'-red, 'g'-green etc.)
    @param PlottingEngine: int 0 = matplotlib, 1 = MATLAB(default)
    @param kwargs:
    @return: return axis handle
    """

    # Draw vectors
    handle = ax.quiver(x, y, z, u, v, w, color=color, length=length, **kwargs)
    plt.draw()
    return handle

def mesh(ax, x, y, z, **kwargs):
    """!
    Displays x, y, z wireframe mesh
    @param ax: axis handle
    @param x: list of x-coordinate values
    @param y: list of y-coordinate values
    @param z: list of z-coordinate values
    @param PlottingEngine: int 0 = matplotlib, 1 = MATLAB(default)
    @param kwargs:
    @return: return axis handle
    """

    # Auto-generate Figure?
    if ax is None:
        if z is None:
            ax = Figure(size=[800, 600], grid=[1, 1], projection='2d')
        else:
            ax = Figure(size=[800, 600], grid=[1, 1], projection='3d')
        ax = ax[0][0]

    # Plot Mesh
    handle = ax.plot_wireframe(x, y, z)
    plt.draw()
    return handle

def patch(ax, vertices, faces=None, facecolors=None, alpha=1.0, color=plt.cm.jet, linewidth=1.0):
    """"
    Displays triangles defined by vertices as a surface (does triangulation internally)
    @param ax: axis handle
    @param vertices: list of vertices
    @param faces: list of faces
    @param facecolors: list of facecolors
    @param alpha: float for transparency
    @param color: color map
    @param linewidth: float for Line width
    @param PlottingEngine: int 0 = matplotlib, 1 = MATLAB(default)
    @return: return axis handle
    """
    
    # Generate faces?
    if faces is None:
        faces = mtri.Triangulation(vertices[0], vertices[1]).triangles
    # Get colors
    if facecolors is not None:
        colors = [(facecolors[face[0]] + facecolors[face[1]] + facecolors[face[2]]) / 3 for face in faces]
        colors = (colors - amin(facecolors)) / (amax(facecolors) - amin(facecolors))

    # Auto-generate Figure?
    if ax is None:
        if z is None:
            ax = Figure(size=[800, 600], grid=[1, 1], projection='2d')
        else:
            ax = Figure(size=[800, 600], grid=[1, 1], projection='3d')
        ax = ax[0][0]

    # Generate faces?
    if faces is None:
        faces = mtri.Triangulation(vertices[0], vertices[1]).triangles

    # Detect color mode
    if facecolors is not None:
        # Plot face collection
        cmap = plt.cm.jet
        triangles = array([array([[vertices[0][face[0]], vertices[1][face[0]], vertices[2][face[0]]],
                                  [vertices[0][face[1]], vertices[1][face[1]], vertices[2][face[1]]],
                                  [vertices[0][face[2]], vertices[1][face[2]], vertices[2][face[2]]]]) for face in
                           faces])
        colors = [(facecolors[face[0]] + facecolors[face[1]] + facecolors[face[2]]) / 3 for face in faces]
        colors = (colors - amin(facecolors)) / (amax(facecolors) - amin(facecolors))
        collection = Poly3DCollection(triangles, facecolors=cmap(colors), alpha=alpha, linewidth=linewidth)
        res = ax.add_collection(collection)

        # Resize axes manually
        xMin = min(min(ax.get_xlim()), amin(vertices[0]))
        xMax = max(max(ax.get_xlim()), amax(vertices[0]))
        yMin = min(min(ax.get_ylim()), amin(vertices[1]))
        yMax = max(max(ax.get_ylim()), amax(vertices[1]))
        zMin = min(min(ax.get_zlim()), amin(vertices[2]))
        zMax = max(max(ax.get_zlim()), amax(vertices[2]))
        ax.set_xlim(xMin, xMax)
        ax.set_ylim(yMin, yMax)
        ax.set_zlim(zMin, zMax)
    elif isinstance(color, str):
        res = ax.plot_trisurf(vertices[0], vertices[1], vertices[2], triangles=faces, alpha=alpha, color=color,
                              linewidth=linewidth)  # , rasterized=True)
    else:
        res = ax.plot_trisurf(vertices[0], vertices[1], vertices[2], triangles=faces, alpha=alpha, cmap=color,
                              linewidth=linewidth)  # , rasterized=True)
    plt.draw()
    return res

def trisurf(ax, vertices, triangles, **kwargs):
    """!
    Displays triangles defined by vertices and triangles as a surface
    @param ax: axis handle
    @param vertices: list of vertices
    @param triangles: list of traingles
    @param PlottingEngine: int 0 = matplotlib, 1 = MATLAB(default)
    @param kwargs:
    @return: return axis handle
    """
    
    colors = [(vertices[2][triangle[0]] + vertices[2][triangle[1]] + vertices[2][triangle[2]]) / 3 for triangle in triangles]
    colors = (colors - amin(colors)) / (amax(colors) - amin(colors))

    # Plot face collection
    cmap = plt.cm.jet
    faces = array([array([ [vertices[0][triangle[0]], vertices[1][triangle[0]], vertices[2][triangle[0]]],
                               [vertices[0][triangle[1]], vertices[1][triangle[1]], vertices[2][triangle[1]]],
                               [vertices[0][triangle[2]], vertices[1][triangle[2]], vertices[2][triangle[2]]] ]) for triangle in triangles])
    collection = Poly3DCollection(faces, facecolors=cmap(colors), **kwargs)
    handle = ax.add_collection(collection)

    # Resize axes manually
    xMin = min(min(ax.get_xlim()), amin(vertices[0]))
    xMax = max(max(ax.get_xlim()), amax(vertices[0]))
    yMin = min(min(ax.get_ylim()), amin(vertices[1]))
    yMax = max(max(ax.get_ylim()), amax(vertices[1]))
    zMin = min(min(ax.get_zlim()), amin(vertices[2]))
    zMax = max(max(ax.get_zlim()), amax(vertices[2]))
    ax.set_xlim(xMin, xMax)
    ax.set_ylim(yMin, yMax)
    ax.set_zlim(zMin, zMax)

    plt.draw()
    return handle

def colorbar(ax, facecolors):
    """!
    Render the colorbar
    @param ax: axis handle
    @param facecolors: list of colors
    @param PlottingEngine: int 0 = matplotlib, 1 = MATLAB(default)
    """

    m = plt.cm.ScalarMappable(cmap=plt.cm.jet)
    m.set_array(facecolors)
    ax.figure.colorbar(m)
    plt.draw()

def grid(ax, on=True):
    """!
    Turn Grid On/Off in the plot
    @param ax: axis handle
    @param on: bool
    @param PlottingEngine: int 0 = matplotlib, 1 = MATLAB(default)
    """

    # Switch Grid on/off
    ax.grid(on)
    plt.draw()

def labels(ax, xLabel, yLabel, zLabel=None):
    """!
    Draw axes labels
    @param ax: axis handle
    @param xLabel: str Label for X-axis
    @param yLabel: str Label for Y-axis
    @param zLabel: str Label for Z-axis
    @param PlottingEngine: int 0 = matplotlib, 1 = MATLAB(default)
    """

    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    if zLabel is not None:
        ax.set_zlabel(zLabel)
    plt.draw()

def legend(ax, labels, colors):
    """!
    Add Legend to the plot
    @param ax: axis handle
    @param labels: list of labels
    @param colors: list f colors
    @param PlottingEngine: int 0 = matplotlib, 1 = MATLAB (default)
    """
    
    legend = []
    for n in range(len(labels)):
        legend.append(patches.Patch(color=colors[n], label=labels[n]))

    ax.legend(handles=legend)
    plt.draw()

def save(ax, filepath='plot.png'):
    savefig(filepath)

def text(ax, pos = (-0.2, 0.79), data = ''):
    ax.text2D(pos[0], pos[1], data, transform=ax.transAxes)

def title(ax, title=''):
    ax.set_title(title)


# Provide handles for capitalized function calls...
def CopyView(ax1, ax2):
    copyview(ax1, ax2)

def Daspect(ax, x=1, y=1, z=1):
    daspect(ax, x, y, z)

def Figure(size=[800, 600], grid=[1, 1], projection='2d', title=None, noThread=False, mixedPlots2D3D=False, visible=True):
    return figure(size, grid, projection, title, noThread, mixedPlots2D3D, visible)

def Plot(ax, x, y=None, z=None, color='b', marker=None):
    return plot(ax, x, y, z, color, marker)

def Surf(ax, x, y, z, facecolors=None, alpha=1.0, cmap=plt.cm.jet, zLabel = ''):
    return surf(ax, x, y, z, facecolors, alpha, cmap, zLabel)

def Scatter(ax, x, y=None, z=None, color='b', marker='o', area=20):
    return scatter(ax, x, y, z, color, marker, area)

def Quiver(ax, x, y, z, u, v, w, color='b', length=1.0, linewidth=1.0):
    return quiver(ax, x, y, z, u, v, w, color, length, linewidth)

def Mesh(ax, x, y, z):
    return mesh(ax, x, y, z)
    
def Patch(ax, vertices, faces=None, facecolors=None, alpha=1.0, color=plt.cm.jet, linewidth=1.0):
    return patch(ax, vertices, faces, facecolors, alpha, color, linewidth)

def ColorBar(ax, facecolors):
    colorbar(ax, facecolors)
    
def Grid(ax, on=True):
    grid(ax, on)

def Labels(ax, xLabel, yLabel, zLabel=None):
    labels(ax, xLabel, yLabel, zLabel)

def Legend(ax, labels, colors):
    legend(ax, labels, colors)

def Save(ax, filepath='plot.png'):
    save(ax, filepath)

def Text(ax, pos = (-0.2, 0.79), data = ''):
    text(ax, pos, data)

def Title(ax, msg=''):
    title(ax, msg)

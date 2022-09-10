
__author__ = "Anthony Beaucamp"
__copyright__ = "Copyright (C) 2020 Kyoto University"
__license__ = "Public Domain"
__version__ = "1.0"

# "Matlab" style functions for interpolation

from numpy import *
from scipy.interpolate import interp1d, griddata
from numpy.polynomial import polynomial

def interp1(xList, yList, xCurve, method='linear'):
    """!
    Interpolate a 1-D function.

    x and y are arrays of values used to approximate some function xCurve
    using interpolation to find the value of new points.

    @param xList: array
    @param yList: array
    @param xCurve: function
    @param method: string indicating the type of interpolation to perform e.g 'Linear'
    """
    removalInterp = interp1d(xList, yList, method)
    return removalInterp(xCurve)
            
def interp2(xGrid, yGrid, zGrid, xCurve, yCurve, method='linear'):
    """!
    Interpolate over a 2-D grid.
    x, y and z are arrays of values used to approximate some function f: z = f(x, y)

    @param xGrid: array
    @param yGrid: array
    @param zGrid: array
    @param xCurve: function
    @param yCurve: function
    @param method: string indicating the type of interpolation to perform e.g 'Linear'
    """
    return griddata((hstack(xGrid),hstack(yGrid)), hstack(zGrid), (xCurve,yCurve), method)

def polyfit2d(x, y, z, deg):
    x = asarray(x)
    y = asarray(y)
    z = asarray(z)
    deg = asarray(deg)
    v = polynomial.polyvander2d(x, y, deg)
    v = v.reshape((-1,v.shape[-1]))
    z = z.reshape((v.shape[0],))
    c = linalg.lstsq(v, z)[0]
    return c.reshape(deg+1)

def polyval2d(x, y, poly):
    return polynomial.polyval2d(x, y, poly)

def meshnorm(vertices, faces):
    """!
    Computes the mesh vertex normals from the face normals.
    @param vertices: vertex data list
    @param faces: face list data
    @return: vertex normals
    """
    vert = transpose(vertices)
    norm = zeros( vert.shape, dtype=vert.dtype )
    #Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vert[faces]
    #Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle             
    n = cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices, 
    # we need to normalize these, so that our next step weights each normal equally.
    lens = sqrt( n[:,0]**2 + n[:,1]**2 + n[:,2]**2 )
    n[:,0] /= lens; n[:,1] /= lens; n[:,2] /= lens     
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle, 
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[ faces[:,0] ] += n
    norm[ faces[:,1] ] += n
    norm[ faces[:,2] ] += n
    lens = sqrt( norm[:,0]**2 + norm[:,1]**2 + norm[:,2]**2 )
    norm[:,0] /= lens; norm[:,1] /= lens; norm[:,2] /= lens      
    return transpose(norm)

def surfnorm(xGrid,yGrid,zGrid):
    """!
    Computes surface normals from x,y,z grid data
    @param xGrid: list of x data
    @param yGrid: list of y data
    @param zGrid: list of z data
    @return: tuple containing the xNormals, yNormal and zNormal components.
    """

    s = shape(zGrid)
    numel = (s[0]) * (s[1])
    dnumel = (s[0]-1) * (s[1]-1)

    xU = diff(xGrid[:, 0:s[1]-1],axis=0)
    yU = diff(yGrid[:, 0:s[1]-1],axis=0)
    zU = diff(zGrid[:, 0:s[1]-1],axis=0)
    tU = concatenate((xU.reshape(dnumel,1), yU.reshape(dnumel,1), zU.reshape(dnumel,1)),1)

    xV = diff(xGrid[0:s[0]-1, :],axis=1)
    yV = diff(yGrid[0:s[0]-1, :],axis=1)
    zV = diff(zGrid[0:s[0]-1, :],axis=1)
    tV = concatenate((xV.reshape(dnumel,1), yV.reshape(dnumel,1), zV.reshape(dnumel,1)),1)

    xI = xGrid[0:s[0]-1, 0:s[1]-1] + xU/2 + xV/2
    yI = yGrid[0:s[0]-1, 0:s[1]-1] + yU/2 + yV/2
    zI = zGrid[0:s[0]-1, 0:s[1]-1] + zU/2 + zV/2

    nI = cross(tU,tV) 
    mag = sqrt(nI[:,0]**2 + nI[:,1]**2 + nI[:,2]**2).reshape(s[0]-1, s[1]-1)
    xNI = nI[:,0].reshape(s[0]-1, s[1]-1) / mag
    yNI = nI[:,1].reshape(s[0]-1, s[1]-1) / mag
    zNI = nI[:,2].reshape(s[0]-1, s[1]-1) / mag

    xN = griddata((hstack(xI),hstack(yI)), hstack(xNI), (hstack(xGrid),hstack(yGrid)), 'cubic').reshape(s[0],s[1])
    xN[isnan(xN)] = griddata((hstack(xI),hstack(yI)), hstack(xNI), (xGrid[isnan(xN)],yGrid[isnan(xN)]), 'nearest')

    yN = griddata((hstack(xI),hstack(yI)), hstack(yNI), (hstack(xGrid),hstack(yGrid)), 'cubic').reshape(s[0],s[1])
    yN[isnan(yN)] = griddata((hstack(xI),hstack(yI)), hstack(yNI), (xGrid[isnan(yN)],yGrid[isnan(yN)]), 'nearest')

    zN = griddata((hstack(xI),hstack(yI)), hstack(zNI), (hstack(xGrid),hstack(yGrid)), 'cubic').reshape(s[0],s[1])
    zN[isnan(zN)] = griddata((hstack(xI),hstack(yI)), hstack(zNI), (xGrid[isnan(zN)],yGrid[isnan(zN)]), 'nearest')
    
    norm = sqrt(xN**2+ yN**2 + zN**2)
    xN /= norm; yN /= norm; zN /= norm; 
    
    return -xN, -yN, -zN
  
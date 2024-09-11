# pylint: disable=C0103
import numpy as np
import os
import matplotlib.pyplot as plt
# from scipy import interpolate
from scipy.stats import linregress
from pyproj import Proj, transform
import yaml

def which_computer():
    computername = os.environ['COMPUTERNAME']
    drive = 'C:/'
    if computername == 'IGSAGIEGWSCSH10':
        drive = 'D:/'
    return drive, computername


def replace_nans_with_x(map, x, isy, iyback):
    '''Replace nans in 2D map between beach and back with x'''
    mapx = np.copy(map)
    (nacross, nalong) = np.shape(mapx)
    # arrays for statistics
    sum_prof = np.zeros(nalong)
    sum_repl = np.zeros(nalong)

    for i in range(nalong):
        # cross-island indices of beach, island back
        ibeach = int(isy[i])     # beach varies with survey
        ibck = int(iyback[i])      # yback is constant
        
        # replace nans with x
        tmp_prof = np.squeeze(map[:,i])
        ireplace = np.argwhere(np.isnan(tmp_prof))
        ireplace = ireplace[ np.where( ireplace >= ibeach) ]
        ireplace = ireplace[ np.where( ireplace <= ibck)]
        sum_prof[i] = (ibck-ibeach)
        sum_repl[i] = len(ireplace)
        tmp_prof[ireplace] = x
        mapx[:,i]=tmp_prof
            
    print('Sum points:', np.nansum(sum_prof))
    print('Sum replaced:', np.nansum(sum_repl))
    print('Fraction replaced:', np.nansum(sum_repl)/np.nansum(sum_prof))
    print('Median number replaced', np.nanmedian(sum_repl))
    print('Max. number replaced', np.nanmax(sum_repl))
    print('Median fraction replaced', np.nanmedian(sum_repl/sum_prof))
    return mapx




def fill_nans(a, fill_val = 0.):
    map_shape = np.shape(a)
    print('Map shape:', map_shape)
    ar = np.ravel(a)
    ireplace = np.argwhere(np.isnan(ar))
    print('Replacing:',len(ireplace))
    ar[ireplace] = fill_val


def nanlsfit(x, y):
    """least-squares fit of data with NaNs"""
    ok = ~np.isnan(x+y)
    xx = x[ok]
    yy = y[ok]
    n = len(xx)
    slope, intercept, r, p, stderr = linregress(xx, yy)
    print("n={}; slope, intercept= {:.4f},{:.4f}; r={:.4f} p={:.4f}, stderr={:.4f} ".format(n, slope, intercept, r, p, stderr))
    return n, slope, intercept, r, p, stderr


def stat_summary(x, iprint=False):
    n = len(x)
    nnan = np.sum(np.isnan(x))
    nvalid = n-nnan
    # intitialize with NaNs

    if n > nnan:
        meanx = np.nanmean(x)
        stdx = np.nanstd(x)
        minx = np.nanmin(x)
        d5 = np.nanpercentile(x, 5.)
        d25 = np.nanpercentile(x, 25.)
        d50 = np.nanpercentile(x, 50.)
        d75 = np.nanpercentile(x, 75.)
        d95 = np.nanpercentile(x, 95.)
        maxx = np.nanmax(x)
    else:
        meanx = np.nan
        stdx = np.nan
        minx = np.nan
        d5 = np.nan
        d25 = np.nan
        d50 = np.nan
        d75 = np.nan
        d95 = np.nan
        maxx = np.nan

    # return it in a dict
    s = {'n':n, 'nnan':nnan, 'nvalid':nvalid, 'mean':meanx, 'std':stdx, 'min':minx, 'max':maxx,
         'd5':d5, 'd25':d25, 'd50':d50, 'd75':d75, 'd95':d95}
    # if iprint:
    #     for key, value in s.items():
    #         print('{:6s} = {:.3f}'.format(key, value)),
    if iprint:
        print("  n, nnan, nvalid: ",s['n'],s['nnan'],s['nvalid'])
        print("  mean, std, min, max   : {:.3f} {:.3f} {:.3f} {:.3f}"
            .format(s['mean'], s['std'], s['min'], s['max']))
        print("  d5, d25, d50, d75, d95: {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}"
            .format(s['d5'], s['d25'], s['d50'], s['d75'], s['d95']))

    return s


def running_mean(y, npts):
    '''
    Smooth a 1-d array with a moving average
    https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way

    Input:
        y - 1-d array
        npts - number of points to average
    Returns:
        ys - smoothed arrays
    '''
    box = np.ones(npts)/npts
    ys = np.convolve(y, box, mode='same')
    return ys


def running_nanmean(y, npts):
    '''
    Smooth a 1-d array with a moving average
    https://stackoverflow.com/questions/40773275/sliding-standard-deviation-on-a-1d-numpy-array

    Input:
        y - 1-d array
        npts - number of points to average
    Returns:
        ys - smoothed arrays
    '''
    sy = np.ones_like(y)*np.nan
    nrows = y.size - npts + 1
    n = y.strides[0]
    y2D = np.lib.stride_tricks.as_strided(y,shape=(nrows,npts),strides=(n,n))
    nclip = int((npts-1)/2)
    # print(nclip)
    sy[nclip:-nclip] = np.nanmean(y2D,1)
    return sy


def running_nanmin(y, npts):
    '''
    Smooth a 1-d array with a moving minimum
    https://stackoverflow.com/questions/40773275/sliding-standard-deviation-on-a-1d-numpy-array

    Input:
        y - 1-d array
        npts - number of points to average
    Returns:
        ys - smoothed arrays
    '''
    sy = np.ones_like(y)*np.nan
    nrows = y.size - npts + 1
    n = y.strides[0]
    y2D = np.lib.stride_tricks.as_strided(y, shape=(nrows, npts), strides=(n, n))
    nclip = int((npts-1) / 2)
    # print(nclip)
    sy[nclip:-nclip] = np.nanmin(y2D, 1)
    return sy


def running_stddev(y, npts):
    """
    Smooth a 1-d array w/ moving average of npts
    Return array of smoothed data and moving std. deviation
    https://stackoverflow.com/questions/40773275/sliding-standard-deviation-on-a-1d-numpy-array

    Input:
        y - 1-d array
        npts - number of points to average
    Returns:
        sy -  array of running std. deviation
    """
    sy = np.ones_like(y)*np.nan
    nrows = y.size - npts + 1
    n = y.strides[0]
    y2D = np.lib.stride_tricks.as_strided(y, shape=(nrows, npts), strides=(n, n))
    nclip = int((npts-1) / 2)
    # print(nclip)
    sy[nclip:-nclip] = np.nanstd(y2D, 1)
    return sy


def centroid(x, z):
    cz = np.nanmean(z)
    cx = np.nansum(z * x) / np.nansum(z)
    return(cx, cz)


def make_grid(name=None, e0=None, n0=None, xlen=None, ylen=None, dxdy=None, theta=None):
    """
    Make a rectangular grid to interpolate elevations onto.
    Takes as argument array of dicts, like:
    r={'name': 'ncorebx_refac', 'e0': 378490., 'n0': 3855740., 'xlen': 36500.0, 'ylen': 1500.0, 'dxdy': 1.0, 'theta': 42.0}
    where:
      e0 - UTM Easting of origin [m]
      n0 - UTM Northing of origin [m]
      xlen - Length of alongshore axis [m]
      ylen - Length of cross-shore axis [m]
      dxdy - grid size (must be isotropic right now) [m]
      theta - rotation CCW from x-axis [deg]
    """
    nx = int((1./dxdy) * xlen)
    ny = int((1./dxdy) * ylen)

    xcoords = np.linspace(0.5*dxdy,xlen-0.5*dxdy,nx)
    ycoords = np.linspace(0.5*dxdy,ylen-0.5*dxdy,ny)

    # these will be the coordinates in rotated space
    xrot, yrot = np.meshgrid(xcoords, ycoords, sparse=False, indexing='xy')

    print('make_grid: Shape of xrot, yrot: ',np.shape(xrot),np.shape(yrot))
    shp = np.shape(xrot)
    xu, yu = box2UTMh(xrot.flatten(), yrot.flatten(), e0, n0, theta)
    xu=np.reshape(xu,shp)
    yu=np.reshape(yu,shp)
    # write the UTM coords of the corners to an ASCII file
    corners = np.asarray(  [[xu[0][0],yu[0][0]],
                           [xu[0][-1],yu[0][-1]],
                           [xu[-1][-1],yu[-1][-1]],
                           [xu[-1][0],yu[-1][0]],
                           [xu[0][0],yu[0][0]]])

    print('corners x, corners y]')
    print(corners)
    fn_corners = name+'.csv'
    print('Saving to '+fn_corners)
    np.savetxt(fn_corners, corners, delimiter=",")
    return xu, yu, xrot, yrot, xcoords, ycoords


def box2UTMh(x, y, x0, y0, theta):
    '''
    2D rotation and translation of x, y
    Input:
        x, y - row vectors of original coordinates (must be same size)
        x0, y0 - Offset (location of x, y = (0,0) in new coordinate system)
        theta - Angle of rotation (degrees, CCW from x-axis == Cartesian coorinates)
    Returns:
        x_r, y_r - rotated, offset coordinates
    '''
    thetar = np.radians(theta)
    c, s = np.cos(thetar), np.sin(thetar)

    # homogenous rotation matrix
    Rh = np.array(((c, -s,  0.),
                   (s,  c,  0.),
                   (0., 0., 1.)))
    # homogenous translation matrix
    Th = np.array(((1., 0., x0),
                   (0., 1., y0),
                   (0., 0., 1.)))

    # homogenous input x,y
    xyh = np.vstack((x,y,np.ones_like(x)))

    # perform rotation and translation
    xyrh = np.matmul(np.matmul(Th,Rh), xyh)
    x_r = xyrh[0,:]
    y_r = xyrh[1,:]
    return x_r, y_r


def pcoord(x, y):
    """
    Convert x, y to polar coordinates r, az (geographic convention)
    r,az = pcoord(x, y)
    """
    r = np.sqrt(x**2 + y**2)
    az = np.degrees(np.arctan2(x, y))
    # az[where(az<0.)[0]] += 360.
    az = (az+360.)%360.
    return r, az


def xycoord(r, az):
    """
    Convert r, az [degrees, geographic convention] to rectangular coordinates
    x,y = xycoord(r, az)
    """
    x = r * np.sin(np.radians(az))
    y = r * np.cos(np.radians(az))
    return x, y


def UTM2Island(eutm, nutm, eoff=383520.0, noff=3860830.0, rot=42.0):
    """
    Convert UTM NAD83 Zone 18N easting, northing to N. Core Banks alongshore, cross-shore coordinates
    xisl, yisl = UTM2Island( eutm, nutm )
    Better to use values from the dict than defaults for translation/rotation values
    Defaults are associated with the dict read in from `small_island_box.yml`
    """
    [r, az] = pcoord(eutm-eoff, nutm-noff)
    az = az + rot
    [xisl,yisl] = xycoord(r,az)
    return xisl, yisl


def island2UTM(alongshore, across_shore, eoff=383520.0, noff=3860830.0, rot=42.):
    """Convert island coordinates to UTM
       Inverse of UTM2Island()
       Better to use values from the dict than defaults for translation/rotation values
       Defaults are associated with the dict read in from `small_island_box.yml`

       Here is code for UTM2island:
          [r, az] = pcoord(eutm-eoff, nutm-noff)
          az = az + rot
          [xisl,yisl] = xycoord(r,az)
    """
    r, az = pcoord(alongshore, across_shore)
    az = az - rot
    eUTM, nUTM = xycoord(r, az)
    eUTM = eUTM + eoff
    nUTM = nUTM + noff
    return eUTM, nUTM


def LatLon2UTM(lat,lon,init_epsg='epsg:26918'):
    """
    Convert lat lon (WGS84) to UTM.
    Defaults to Zone 18N

    TODO: Update to Proj 6 and correct this syntax
    """
    inProj = Proj(init='epsg:4326')
    outProj = Proj(init=init_epsg)
    outx,outy = transform(inProj,outProj,lon,lat)
    return outx, outy


def UTM2LatLon(easting, northing, initepsg='epsg:26918'):
    """
    Convert UTM to lat, lon (WGS84)
    Defaults to Zone 18N

    TODO: Update to Proj 6 and correct this syntax
    """
    outProj = Proj(init='epsg:4326')
    inProj = Proj(init=initepsg)
    lon,lat = transform(inProj,outProj,easting,northing)
    return lon, lat


def UTM2rot(xutm,yutm,r):
    """
    Convert UTM coordinates to rotated coordinates

    Now deprecated by UTM2Island ... delete
    """
    # Convert origin to UTM
    xu,yu = box2UTMh(0.,0.,r['e0'],r['n0'],r['theta'])
    # reverse the calc to find the origin (UTM =0,0) in box coordinates.
    # First, just do the rotation to see where Box = 0,0 falls
    xb0,yb0 = box2UTMh(xu,yu,0.,0.,-r['theta'])
    # Then put in negative values for the offset
    #TODO: why does this return a list of arrays?
    xbl,ybl = box2UTMh(xutm,yutm,-xb0,-yb0,-r['theta'])
    # this fixes it...probably should fix box2UTMh
    xb = np.concatenate(xbl).ravel()
    yb = np.concatenate(ybl).ravel()
    return xb, yb


def yaml2dict(yamlfile):
    """Import contents of a YAML file as a dict

    Args:
        yamlfile (str): YAML file to read

    Returns:
        dict interpreted from YAML file

    Raises:

    """
    dictname = None
    with open(yamlfile, "r") as infile:
        try:
            dictname = yaml.safe_load(infile)
        except yaml.YAMLError as exc:
            print(exc)

    return dictname


def map_stats(mp,sfile):
    '''
    Calculate some basic statistics for 3D map arrays
    '''
    mean = np.nanmean(mp,axis=(1,2))
    mad = np.nanmean(np.abs(mp),axis=(1,2))
    dmin = np.nanmin(mp,axis=(1,2))
    dmax = np.nanmax(mp,axis=(1,2))
    rms = np.sqrt(np.nanmean(mp**2.,axis=(1,2)))
    s = np.shape(mp)
    num = []
    numn = []
    for i in range(s[0]):
        num.append(mp[i,:,:].size)
        numn.append(np.count_nonzero(np.isnan(mp[i,:,:])))
    print("Shape: ",s,file=sfile)
    print("mean",mean,file=sfile)
    print("mad",mad,file=sfile)
    print("min",dmin,file=sfile)
    print("max",dmax,file=sfile)
    print("rms",rms,file=sfile)
    print("nans",numn,file=sfile)
    print("size",num,file=sfile)
    return mean, mad


def map_stats2d(mp,sfile):
    '''
    Calculate some basic statistics for 2D map arrays
    '''
    mean = np.nanmean(mp,axis=(0,1))
    mad = np.nanmean(np.abs(mp),axis=(0,1))
    dmin = np.nanmin(mp,axis=(0,1))
    dmax = np.nanmax(mp,axis=(0,1))
    rms = np.sqrt(np.nanmean(mp**2.,axis=(0,1)))
    s = np.shape(mp)
    num = (mp[:,:].size)
    numn = (np.count_nonzero(np.isnan(mp[:,:])))
    print("Shape: ",s,file=sfile)
    print("mean",mean,file=sfile)
    print("mad",mad,file=sfile)
    print("min",dmin,file=sfile)
    print("max",dmax,file=sfile)
    print("rms",rms,file=sfile)
    print("nans",numn,file=sfile)
    print("size",num,file=sfile)
    return mean, mad

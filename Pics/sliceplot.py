#!/usr/bin/env python
# SIMPLEST USAGE: python sliceplot.py -i file1 file2 file3...
#
# More complex options:
USAGE = "USAGE: python sliceplot.py [--savearray] [--savefig] [--zindex=<z-index of slice>] [--delzindex=<offset for subtracting image>] [--filter=<smoothing sigma>] [--filterx=<x-axis smoothing sigma>] [--filtery=<y-axis smoothing sigma>] [--filterz=<z-axis smoothing sigma>] [--min=<min of plot>] [--max=<max of plot>] -i '<filename1> <filename2>...'"

###### LIST OF OPTIONAL ARGUMENTS:
###  --savearray is a flag indicating you want to also save the 2D slice as a .npy file.
###  --savefig lets you save the sliceplot figure
###  --zindex= lets you specify which array index cut through the z axis (usualy the LOS axis).  DEFAULT is the midpoint, i.e. DIM/2
###  --delzindex= if this is specified, then we will plot the difference between slices at array[:,:,zindex] - array[:,:,zindex+delzindex]
###  --filter= allows you to smooth the array with a Gaussian filter with the specified standard deviation (in units of array cells).  DEFAULT is no smoothing.
###  --filterx= smooth only along the horizontal axis.  DEFAULT is no smoothing.
###  --filtery= smooth only along the vertical axis.  DEFAULT is no smoothing.
###  --filterz= smooth only along the line of sight axis.  DEFAULT is no smoothing.
###  --min= specify a minimum value for the plotted range
###  --max= specify a maximum value for the plotted range

from matplotlib.mlab import griddata
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import scipy as sp
# from scipy import ndimage
from matplotlib.ticker import *
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from os.path import basename
import os
import sys, getopt


#To normalize the midpoint of the colorbar
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def load_binary_data(filename, dtype=np.float32):
     """
     We assume that the data was written
     with write_binary_data() (little endian).
     """
     f = open(filename, "rb")
     data = f.read()
     f.close()
     _data = np.fromstring(data, dtype)
     if sys.byteorder == 'big':
       _data = _data.byteswap()
     return _data


#default arguments
x_sigma = -1 # if negative, then do not smooth the field
y_sigma = -1
z_sigma = -1
iso_sigma = -1
files_in = []
z_index = -1
minrange = 1e5
maxrange = -1e5
crop_size = 0
save_array = 0
save_fig = 0
del_z_index = int(0)

# check for optional arguments
if(1):
    try:
        opts, args = getopt.getopt(sys.argv[1:],"u:f:x:z:y:i:", ["filter=", "filterx=", "filtery=", "filterz=", "fx=", "fy=", "fz=", "zindex=", "min=", "max=", "savearray", "savefig", "delzindex="])
    except getopt.GetoptError:
        print USAGE
        sys.exit(2)
    for opt, arg in opts:
        #    print opt,arg
        if opt in ("-u", "--u", "-h", "--h", "--help"):
            print USAGE
            sys.exit()
        elif opt in ("-x", "-filterx", "--filterx"):
            x_sigma = float(arg)
        elif opt in ("-y", "-filtery", "--filtery"):
            y_sigma = float(arg)
        elif opt in ("-z", "-filterz", "--filterz"):
            z_sigma = float(arg)
        elif opt in ("-f", "--f", "-filter", "--filter"):
            iso_sigma = float(arg)
        elif opt in ("-i", "--i"):
            files_in = arg
            files_in = files_in.split()
        elif opt in ("-zindex", "--zindex"):
            z_index = int(arg)
        elif opt in ("-delzindex", "--delzindex"):
            del_z_index = int(arg)
        elif opt in ("-min", "--min"):
            minrange = float(arg)
        elif opt in ("-max", "--max"):
            maxrange = float(arg)
        elif opt in ("--crop="):
            crop_size = int(arg)
        elif opt in ("--savearray"):
            save_array = 1
        elif opt in ("--savefig"):
            save_fig = 1

if not files_in:
    print "No files for processing... Have you included a '-i' flag before the filenames?\n"+USAGE
    sys.exit()


def is_Fcoll(data, dim):
    """
    Check if the data is the Fcoll array based on its length
    """
    if len(data) == dim**2 * (dim + 2):
        return True
    elif len(data) == dim**3:
        return False
    else:
        raise ValueError("Dimensions of data do not match known types")

def reshape_data(data, dim):
    """
    Reshapes the data depending on its dimensions
    """
    if is_Fcoll(data, dim):
        data.shape = (dim, dim, dim + 2)
        data = data[:,:,:-2]  # slice so same size as other boxes
    else:
        data.shape = (dim, dim, dim)
    return data

def smooth_field(data, iso_sigma, x_sigma, y_sigma, z_sigma):
    """
    Smooth the field if any of the smoothing parameters are positive
    """
    if iso_sigma > 0:
        print "Smoothing the entire cube with a Gassian filter of width="+str(iso_sigma)
        data =  sp.ndimage.filters.gaussian_filter(data, sigma=iso_sigma)
        # k=Create_Kernel(iso_sigma)
        # data = sp.ndimage.convolve(data, k, mode='constant', cval=0.0)
    else:
        if x_sigma > 0:
            print "Smoothing along the x (horizontal) axis with a Gassian filter of width="+str(x_sigma)
            data =  sp.ndimage.filters.gaussian_filter1d(data, sigma=x_sigma, axis=1)
        if y_sigma > 0:
            print "Smoothing along the y (vertical) axis with a Gassian filter of width="+str(y_sigma)
            data = sp.ndimage.filters.gaussian_filter1d(data, sigma=y_sigma, axis=0)
        if z_sigma > 0:
            print "Smoothing along the z (line of sight) axis with a Gassian filter of width="+str(z_sigma)
            data = sp.ndimage.filters.gaussian_filter1d(data, sigma=z_sigma, axis=2)
    return data

def crop(data, length):
    """
    Crop the box to a size given by length (int)
    """
    if length > 0:
        return data[:length,:length,:length]
    else:
        return data

def Create_Kernel(radius):
    '''
    Creates 3d spherical kernel. Radius converted to int (rounds down) and made positive.
    '''
    radius=int(np.abs(radius))

    kernel_matrix = np.zeros((2*radius+1, 2*radius+1, 2*radius+1))
    x,y,z = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 + z**2 <= radius**2
    kernel_matrix[mask] = 1
    kernel_matrix=kernel_matrix/kernel_matrix.sum()

    return kernel_matrix


# go through list of files and process each one
for path in files_in:
    #path = sys.argv[i]
    print 'Processing input file:'
    print '  ' + path
    filename = "" + path.split("/")[-1]

    # lightcone?
    if basename(filename)[-11:] == 'lighttravel':
        DIM = int("" + path.split("_")[-3])
        label = str("" + path.split("_")[-2])

    else:
        DIM = int("" + path.split("_")[-2])
        label = str("" + path.split("_")[-1])

    if z_index < 0:
        z_index = DIM/2

    data1 = load_binary_data(path)

    data1 = reshape_data(data1, DIM)
    data1 = smooth_field(data1, iso_sigma, x_sigma, y_sigma, z_sigma)
    data1 = crop(data1, crop_size)

    fig = plt.figure(dpi=72)
    sub_fig = fig.add_subplot(111)
    print "Taking a slice along the LOS direction at index="+str(z_index)
    the_slice = data1[:,:,z_index]

    if del_z_index: #difference image is wanted
        other_z_index = int(z_index+del_z_index)
        print "Subtracting the slice at index="+str(other_z_index)
        the_slice = the_slice - data1[:,:,other_z_index]

    ax = plt.subplot()

    # check box type to determine default plotting options
    # check if it is a 21cm brightness temperature box
    if basename(filename)[0:3] == 'del':
        if minrange > 1e4:
            minrange = -210
        if maxrange < -1e4:
            maxrange = 30
        # cmap = LinearSegmentedColormap.from_list('mycmap', ['yellow','red','black','green','blue'])
        cmap = LinearSegmentedColormap.from_list('mycmap', ['cyan','blue','black','red','yellow'])
        norm = MidpointNormalize(midpoint=0)
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_ticks([])
        frame1.axes.get_yaxis().set_ticks([])
        # frame1.set_xlabel(r'${\rm\longleftarrow %s \longrightarrow}$'%(label), fontsize=20)
        c_dens = sub_fig.imshow(the_slice,cmap=cmap,norm=norm)
        c_dens.set_clim(vmin=minrange,vmax=maxrange)
        c_bar = fig.colorbar(c_dens, orientation='vertical')
        c_bar.set_label(r'${\rm \delta T_b (\mathrm{mK})}$', fontsize=16, rotation=-90, labelpad=32)
        tick_array = np.linspace(minrange, maxrange, 8)
        plt.title("21 cm brightness temperature", fontproperties=font_prop, size=16, verticalalignment='bottom')
        fig_name = "del_T z7"

    # check if it is a neutral fraction box
    elif basename(filename)[0:3] == 'xH_':
        if minrange > 1e4:
            minrange = 0
        if maxrange < -1e4:
            maxrange = 1
        cmap = LinearSegmentedColormap.from_list('mycmap', ['black','white'])
        norm = MidpointNormalize(midpoint=0.5)
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_ticks([])
        frame1.axes.get_yaxis().set_ticks([])
        frame1.set_xlabel(r'${\rm\longleftarrow %s \longrightarrow}$'%(label), fontsize=20)
        c_dens = sub_fig.imshow(the_slice,cmap=cmap,norm=norm)
        c_dens.set_clim(vmin=minrange,vmax=maxrange)
        c_bar = fig.colorbar(c_dens, orientation='vertical')
        c_bar.set_label(r'${\rm x_{HI}}$', fontsize=16, rotation=-90, labelpad=32)
        tick_array = np.linspace(minrange, maxrange, 6)
        # plt.title("Neutral fraction", size=16, verticalalignment='bottom')
        fig_name = "xH z7"

    # check it is a density box
    elif basename(filename)[0:3] == 'upd':
        if minrange > 1e4:
            minrange = -0.5
        if maxrange < -1e4:
            maxrange = 0.5
        the_slice = np.log10(1+data1[:,:,z_index])
        cmap = LinearSegmentedColormap.from_list('mycmap', ['darkblue', 'blue', 'cyan', 'yellow', 'red', 'darkred'])
        norm = MidpointNormalize(midpoint=0)
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_ticks([])
        frame1.axes.get_yaxis().set_ticks([])
        frame1.set_xlabel(r'${\rm\longleftarrow %s \longrightarrow}$'%(label), fontsize=20)
        c_dens = sub_fig.imshow(the_slice,cmap=cmap,norm=norm)
        c_dens.set_clim(vmin=minrange,vmax=maxrange)
        c_bar = fig.colorbar(c_dens, orientation='vertical')
        c_bar.set_label(r'${\rm log(\Delta)}$', fontsize=16, rotation=-90, labelpad=32)
        tick_array = np.linspace(minrange, maxrange, 5)
        # plt.title("Density", size=16, verticalalignment='bottom')
        fig_name = "density z7"

    # check it is an Fcoll box
    elif basename(filename)[0:18] == 'Fcoll_output_file_':
        # the_slice = np.log10(1 + the_slice)
        if minrange > 1e4:
            minrange = 0.
        if maxrange < -1e4:
            maxrange = 1.
        cmap = LinearSegmentedColormap.from_list('mycmap', ['black', 'red', 'yellow', 'white', 'white', 'white'])
        norm = MidpointNormalize(midpoint=maxrange/2.)
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_ticks([])
        frame1.axes.get_yaxis().set_ticks([])
        frame1.set_xlabel(r'${\rm\longleftarrow %s \longrightarrow}$'%("300Mpc"), fontsize=20)
        c_dens = sub_fig.imshow(the_slice,cmap=cmap,norm=norm)
        c_dens.set_clim(vmin=minrange,vmax=maxrange)
        c_bar = fig.colorbar(c_dens, orientation='vertical')
        # c_bar.set_label(r'${\rm log(f_{coll})}$', fontsize=24, rotation=-90, labelpad=32)
        c_bar.set_label(r'${\rm f_{coll}}$', fontsize=16, rotation=-90, labelpad=32)
        tick_array = np.linspace(minrange, maxrange, 5)
        # plt.title("Collapse fraction", size=16, verticalalignment='bottom')
        fig_name = "fcoll z7"


    c_bar.set_ticks(tick_array)

    for t in c_bar.ax.get_yticklabels():
        t.set_fontsize(14)

    if del_z_index:
        endstr = '_zindex'+str(z_index)+'-'+str(z_index+del_z_index)
    else:
        endstr = '_zindex'+str(z_index)
    # plt.savefig(filename+endstr+'.png', bbox_inches='tight')

    if save_fig:
        plt.savefig(fig_name + ".png")#, dpi=200)
    else:
        plt.show()

    # do we want to save the array file?
    if save_array:
        np.save(filename+endstr, the_slice)

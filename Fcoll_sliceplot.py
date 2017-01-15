USAGE = "USAGE (IPython): run Fcoll_sliceplot.py -i <filename1> [<filename2>...]"

from matplotlib.mlab import griddata
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import scipy as sp
from scipy import ndimage
from matplotlib.ticker import *
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from os.path import basename
import os
import sys, getopt

class MidpointNormalize(Normalize):
    """
    Normalise the midpoint of the colourbar
    """
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
iso_sigma = 0.8
files_in = []
z_index = -1
minrange = 0.
maxrange = 0.05
savefile = 0
del_z_index = int(0)

try:
    opts, args = getopt.getopt(sys.argv[1:], "u:f:x:z:y:i:")
except getopt.GetoptError:
    print USAGE
    sys.exit(2)

for opt, arg in opts:
    if opt in ("-u", "--u", "-h", "--h", "--help"):
        print USAGE
        sys.exit()
    elif opt in ("-i", "--i"):
        files_in = arg
        files_in = files_in.split()

if not files_in:
    print "No files for processing... Have you included a '-i' flag before the filenames?\n" + USAGE
    sys.exit()

# Go through list of files and process each one:
for path in files_in:
    #path = sys.argv[i]
    print 'Processing input file:'
    print '  '+path
    filename="" + path.split("/")[-1]

    # DIM = int("" + path.split("_")[-2])
    # label=str("" + path.split("_")[-1])
    DIM = 256

    if z_index < 0:
        z_index = DIM/2

    # Read in the data cube:
    Fcoll = load_binary_data(path)
    Fcoll.shape = (DIM, DIM+2, DIM)
    Fcoll = Fcoll.reshape((DIM, DIM+2, DIM), order='F')

    if iso_sigma > 0:
        print "Smoothing the entire cube with a Gassian filter of width=" + str(iso_sigma)
        Fcoll = sp.ndimage.filters.gaussian_filter(Fcoll, sigma=iso_sigma)


    fig = plt.figure(dpi=72)
    sub_fig = fig.add_subplot(111)
    print "Taking a slice along the LOS direction at index="+str(z_index)
    slice = Fcoll[:,:,z_index]


    if minrange > 1e4:
        minrange = -0.5
    if maxrange < -1e4:
        maxrange = 0.5
    slice = np.log10(1 + Fcoll[:, :, 250])
    cmap = LinearSegmentedColormap.from_list('mycmap', ['black', 'red', 'yellow', 'white'])
    norm = MidpointNormalize(midpoint=maxrange/2.)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_ticks([])
    frame1.axes.get_yaxis().set_ticks([])
    frame1.set_xlabel(r'${\rm\longleftarrow %s \longrightarrow}$'%("300Mpc"), fontsize=20)
    c_dens = sub_fig.imshow(slice,cmap=cmap,norm=norm)
    c_dens.set_clim(vmin=minrange,vmax=maxrange)
    c_bar = fig.colorbar(c_dens, orientation='vertical')
    c_bar.set_label(r'${\rm log(\Delta)}$', fontsize=24, rotation=-90, labelpad=32)
    tick_array = np.linspace(minrange, maxrange, 5)

    plt.show()

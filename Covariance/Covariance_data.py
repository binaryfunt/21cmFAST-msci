USAGE = "USAGE (IPython): run Fcoll_sliceplot.py [--filter=<smoothing sigma>] -i <Fcoll_output_file> -i <other_box>"

#The binary file containing the Fcoll has to be called "Fcoll_output_file" exactly in the input.
#My example input "run Covariance_data.py -i ../21cmFAST_Fcoll_fullsim/Run1_z9/Fcoll_output_file -i ../21cmFAST_Fcoll_fullsim/Run1_z9/Boxes/xH_nohalos_z009.00_nf0.767682_eff20.0_effPLindex0.0_HIIfilter1_Mmin4.9e+08_RHIImax20_256_300Mpc"
#haven't tested the -f inputs

import numpy as np
import scipy as sp
from scipy import ndimage
import matplotlib.pyplot as plt
from os.path import basename
import sys, getopt
import os

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

Fcoll_iso_sigma=0.8
iso_sigma = 0.8 #default adjusted by -f in input
files_in = []
savefile = 0
del_z_index = int(0)

try:
    opts, args = getopt.getopt(sys.argv[1:], "u:f:x:z:y:i:", ["filter=", "max="])
except getopt.GetoptError:
    print USAGE
    sys.exit(2)

for opt, arg in opts:
    if opt in ("-u", "--u", "-h", "--h", "--help"):
        print USAGE
        sys.exit()
    elif opt in ("-f", "--f", "-filter", "--filter"):
      iso_sigma = float(arg)
    elif opt in ("-i", "--i"):
        files_in.append(arg)
        #files_in = files_in.split()

if not files_in:
    print "No files for processing... Have you included a '-i' flag before the filenames?\n" + USAGE
    sys.exit()

for path in files_in:
    #path = sys.argv[i]
    print 'Processing input file:'
    print '  '+path
    filename="" + path.split("/")[-1]

    DIM = 256

    # Read in the data cube:
    if filename == "Fcoll_output_file":
        Fcoll = load_binary_data(path)
        Fcoll.shape = (DIM, DIM, DIM+2)
        # Fcoll = Fcoll.reshape((DIM, DIM, DIM+2), order='F')
        # Slice so same size as other boxes:
        Fcoll= Fcoll[:,:,:-2]   #slicing so same size as other boxes
        # TODO: find out whether it should be Fcoll[:,:,1:-1] instead

        #print "Fcoll shape", Fcoll.shape

        if Fcoll_iso_sigma > 0:
            print "Smoothing the entire Fcoll box with a Gassian filter of width=" + str(Fcoll_iso_sigma)
            Fcoll = sp.ndimage.filters.gaussian_filter(Fcoll, sigma=Fcoll_iso_sigma)

    else:
        data = load_binary_data(path)
        data.shape = (DIM, DIM, DIM)
        data = data.reshape((DIM, DIM, DIM), order='F')
        #print "data shape", data.shape

        if iso_sigma > 0
            print "Smoothing the entire other box with a Gassian filter of width=" + str(iso_sigma)
            data = sp.ndimage.filters.gaussian_filter(data, sigma=iso_sigma)

def Find_Covariance(box1, box2):
    '''
    Finds covariance matrix between the two boxes.
    '''
    #flatten box arrays to vectors first as np.cov only accepts vectors
    box1_f=box1.flatten()
    box2_f=box2.flatten()

    cov_matrix=np.cov(box1_f,box2_f)

    return cov_matrix


def Find_R(cov_matrix):
    '''
    Finds Pearson's R from covariance matrices.
    '''
    Pearson_R=cov_matrix[0,1]/((cov_matrix[0,0]*cov_matrix[1,1])**0.5)

    return Pearson_R

cov=Find_Covariance(Fcoll, data)
R=Find_R(cov)
print "Covariance Matrix=\n", cov
print "R=", R

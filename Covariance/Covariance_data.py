USAGE = "USAGE (IPython): run Fcoll_sliceplot.py [--filter=<smoothing sigma>] [--max=<max of plot>] -i <filename1> [<filename2>...]"

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

Fcoll_iso_sigma=0.8  #default for the fcoll boxes
iso_sigma = 0. #for the other boxes
files_in = []
z_index = -1
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
      Fcoll_iso_sigma = float(arg)
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
    
    if z_index < 0:
        z_index = DIM/2
    
    # Read in the data cube:
    if filename=="Fcoll_output_file":
        Fcoll = load_binary_data(path)
        Fcoll.shape = (DIM, DIM, DIM+2)
        Fcoll = Fcoll.reshape((DIM, DIM, DIM+2), order='F')
        Fcoll= Fcoll[:,:,:-2]   #slicing so same size as other boxes
        #print "Fcoll shape", Fcoll.shape
        
        if Fcoll_iso_sigma > 0:
            print "Smoothing the entire Fcoll cube with a Gaussian filter of width=" + str(Fcoll_iso_sigma)
            Fcoll = sp.ndimage.filters.gaussian_filter(Fcoll, sigma=Fcoll_iso_sigma)
            
    else:
        data = load_binary_data(path)
        data.shape = (DIM, DIM, DIM)
        data = data.reshape((DIM, DIM, DIM), order='F')
        #print "data shape", data.shape
        
        if iso_sigma > 0:
            print "Smoothing the entire other cube with a Gaussian filter of width=" + str(iso_sigma)
        data = sp.ndimage.filters.gaussian_filter(data, sigma=iso_sigma)

def Find_Covariance(box1, box2):
    '''
    Finds covariance matrix between the two boxes.
    '''
    #flatten box arrays to vectors first
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

#f=open('data3.txt','a')
#f.write(str(R)+"\n")

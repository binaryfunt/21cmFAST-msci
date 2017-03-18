# -*- coding: utf-8 -*-
import numpy as np
from numpy import linalg
import scipy.stats as st
import sys
import time

def load_binary_data(filename, dtype=np.float32):
     """
     We assume that the data was written with write_binary_data() (little endian).
     """
     f = open(filename, "rb")
     data = f.read()
     f.close()
     _data = np.fromstring(data, dtype)
     if sys.byteorder == 'big':
       _data = _data.byteswap()
     return _data

def write_binary_data(filename, data, dtype=np.float32):
     """
     Write binary data to a file and make sure it is always in little endian format.
     """
     f = open(filename, "wb")
     _data = np.asarray(data, dtype)
     if sys.byteorder == 'big':
         _data = _data.byteswap()
     f.write(_data)
     f.close()

def Gaussian_3D(coords, centre, width):
    '''
    Takes grid (coords) as arg, along with centre and width of Gaussian. Returns another grid.
    '''
    normal=[]
    power=0.

    for i in range(3):
        normal.append(1./(width[i]*(2*np.pi)**0.5))
        power += ((coords[i] - centre[i])/width[i])**2

    normal = linalg.norm(normal)
    result = normal*np.exp(-0.5*power)

    return result

class Fcoll_pdf(st.rv_continuous):
    def _pdf(self, x, p, q, r):
        '''
        PDF of fcoll heights
            sum of exp and uniform dist.
        x: random variable
        p,q,r: params set from distribution of fcoll heights from 21cmFAST
        '''
        norm = 1./((p/r)*(1. - np.exp(-r)) + 1./(1.-q))
        return norm*np.exp(-x) + 1./(1.-q)



#create grid
grid_length = 64
x_ = np.linspace(0., grid_length - 1., grid_length)
y_ = np.linspace(0., grid_length - 1., grid_length)
z_ = np.linspace(0., grid_length - 1., grid_length)
x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')

#create data
N_peaks = 500
centre_list = np.random.uniform(0., grid_length, (N_peaks,3))
height_list = np.random.uniform(0., 1., N_peaks)

data = np.zeros((int(grid_length), int(grid_length), int(grid_length)))

start=time.time()

print "Generating Fcoll box:", N_peaks, "peaks"

for i in xrange(N_peaks):
    data += height_list[i] * Gaussian_3D(np.array([x,y,z]), centre_list[i], (2.,2.,2.))
    if i % 5 == 0:
        print i, "peaks generated",
        sys.stdout.flush()
        print "\r",

end=time.time()
print end - start

outputfile = write_binary_data('C:/Users/Ronnie/Documents/21cmFAST-msci/Boxes/Fcoll_output_file_CUBE_z008.10_64_75Mpc', data)

#data1=load_binary_data('Fcoll_output_file')
#print data1.shape
#print data1

import numpy as np
from numpy import linalg
from scipy.stats import multivariate_normal
from scipy.stats import rv_continuous
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

def is_pos_def(x):
    '''
    Checks matrix to see if positive definite.
    '''
    return np.all(np.linalg.eigvals(x) > 0)

class Fcoll_pdf(rv_continuous):
    def _pdf(self, x):
        '''
            Piecewise PDF of fcoll heights
            sum of exp and uniform dist.
            x: random variable
            p, q, r: params set from distribution of fcoll heights from 21cmFAST
            end from analysis of fcoll maps
        '''
        end = 1. - 0.0058135 #found from looking at fraction of points in fcoll map that was equal to 1.
        p = 0.183288865538 #found from scipy.optimise
        q = 1261.4820295
        r = -2.39015541247
        norm = 1./(p/(r + 1.) + q*(1. - end)) #normalisation factor
        return np.piecewise(x, [x<end, x>=end], [lambda x: norm*p*x**r, lambda x: norm*p*x**r + q])

#create grid
grid_length = 64
x_ = np.linspace(0., grid_length - 1., grid_length)
y_ = np.linspace(0., grid_length - 1., grid_length)
z_ = np.linspace(0., grid_length - 1., grid_length)
x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
pos = np.stack((x, y, z), axis = -1)

#create data
N_peaks = 200
centre_list = np.random.uniform(0., grid_length, (N_peaks, 3))
height_list = np.random.uniform(0.8, 1., N_peaks)
height_list_pdf = Fcoll_pdf(a=0., b=1., name='Fcoll_pdf')
height_list = height_list_pdf.pdf(height_list)
rand_matrix_list = np.random.uniform(1., 10., (N_peaks, 3, 3))
data = np.zeros((int(grid_length), int(grid_length), int(grid_length)))

start=time.time()

print "Generating Fcoll box:", N_peaks, "peaks"
for i in xrange(N_peaks):
    # Make positive-definite covariance matrix:
    cov_matrix = np.dot(rand_matrix_list[i],rand_matrix_list[i].transpose())
    #cov_matrix = [[1., 0., 0.], [0., 1., 0.],[0., 0., 1.]]
    data += height_list[i] * multivariate_normal.pdf(pos, centre_list[i], cov_matrix, allow_singular=True)
    if i % 5 == 0:
        print i, "peaks generated",
        sys.stdout.flush()
        print "\r",

#data=data*50./np.sum(data)

end=time.time()
print "time taken", end - start

outputfile = write_binary_data('Boxes/Fcoll_output_file_CUBE_z008.00_64_75Mpc', data)

#data1=load_binary_data('Fcoll_output_file')
#print data1.shape
#print data1

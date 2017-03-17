import numpy as np
import numpy.random as nr
import scipy as sp
import scipy.ndimage


def Create_Kernel(radius):
    '''
    Creates 3d spherical kernel. Radius converted to int and made positive.
    '''
    radius=int(np.abs(radius))
    
    kernel_matrix = np.zeros((2*radius+1, 2*radius+1, 2*radius+1))
    x,y,z = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 + z**2 <= radius**2
    kernel_matrix[mask] = 1
    kernel_matrix=kernel_matrix/kernel_matrix.sum()

    return kernel_matrix
    
k=Create_Kernel(2)
print k

#testing sp.ndimage.convolve

test=nr.randint(0, 2, (5,5,5))*1.
print test

result= sp.ndimage.convolve(test, k, mode='constant', cval=0.0)
print result
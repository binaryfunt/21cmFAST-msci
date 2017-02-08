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

def files_in_func(files_in, num, gaussian_sigma=0., th_size=0., sphere_radius=0.):
    '''
    Reads in the data boxes using load_binary_data.
    '''
    path=files_in[num]
    #print 'Processing input file:'
    #print '  '+path
    filename="" + path.split("/")[-1]
    
    DIM = 256
    
    # Read in the data cube:
    if filename=="Fcoll_output_file":
        Fcoll_array = load_binary_data(path)
        Fcoll_array.shape = (DIM, DIM, DIM+2)
        Fcoll_array = Fcoll_array.reshape((DIM, DIM, DIM+2), order='F')
        Fcoll_array= Fcoll_array[:,:,:-2]   #slicing so same size as other boxes
        
        if gaussian_sigma > 0 and th_size > 0:
            print "both smoothings > 0"
            exit()
        
        elif gaussian_sigma > 0:
            #print "Smoothing the entire Fcoll cube with a Gaussian filter of width=" + str(gaussian_sigma)
            Fcoll_array = sp.ndimage.filters.gaussian_filter(Fcoll_array, sigma=gaussian_sigma)
        
        elif th_size > 0:
            #print "Smoothing the entire Fcoll cube with a Uniform filter of width=" + str(th_size)
            Fcoll_array = sp.ndimage.filters.uniform_filter(Fcoll_array, size=th_size)
        
        elif sphere_radius > 0:
            print "Smoothing the entire Fcoll cube with a Spherical filter of width=" + str(sphere_radius)
            k=Create_Kernel(sphere_radius)
            Fcoll_array = sp.ndimage.convolve(Fcoll_array, k, mode='constant', cval=0.0)
            
        #else:
        #    print "Fcoll box not smoothed."
        
        return Fcoll_array
        
    else:
        otherbox_array = load_binary_data(path)
        otherbox_array.shape = (DIM, DIM, DIM)
        otherbox_array = otherbox_array.reshape((DIM, DIM, DIM), order='F')
        
        if gaussian_sigma > 0 and th_size > 0:
            print "both smoothings > 0"
            exit()
        
        elif gaussian_sigma > 0:
            #print "Smoothing the entire other cube with a Gaussian filter of width=" + str(gaussian_sigma)
            otherbox_array = sp.ndimage.filters.gaussian_filter(otherbox_array, sigma=gaussian_sigma)
        
        elif th_size > 0:
            #print "Smoothing the entire other cube with a Uniform filter of width=" + str(th_size)
            otherbox_array = sp.ndimage.filters.uniform_filter(otherbox_array, size=th_size)
        
        elif sphere_radius > 0:
            print "Smoothing the entire other cube with a Spherical filter of width=" + str(sphere_radius)
            k=Create_Kernel(sphere_radius)
            otherbox_array = sp.ndimage.convolve(otherbox_array, k, mode='constant', cval=0.0)
        
        #else:
        #    print "Other box not smoothed."
        
        return otherbox_array

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

#File paths
nf_files_in=['/Users/tristan2/Desktop/Imperial/Year_4/Project/21cmFAST_Fcoll_fullsim/Run2_z12/Fcoll_output_file', '/Users/tristan2/Desktop/Imperial/Year_4/Project/21cmFAST_Fcoll_fullsim/Run2_z12/Boxes/xH_nohalos_z012.00_nf0.955289_eff20.0_effPLindex0.0_HIIfilter1_Mmin3.3e+08_RHIImax20_256_300Mpc']
density_files_in=['/Users/tristan2/Desktop/Imperial/Year_4/Project/21cmFAST_Fcoll_fullsim/Run2_z12/Fcoll_output_file', '/Users/tristan2/Desktop/Imperial/Year_4/Project/21cmFAST_Fcoll_fullsim/Run2_z12/Boxes/updated_smoothed_deltax_z012.00_256_300Mpc']
Tb_files_in=['/Users/tristan2/Desktop/Imperial/Year_4/Project/21cmFAST_Fcoll_fullsim/Run2_z12/Fcoll_output_file', '/Users/tristan2/Desktop/Imperial/Year_4/Project/21cmFAST_Fcoll_fullsim/Run2_z12/Boxes/delta_T_v3_no_halos_z012.00_nf0.955289_useTs0_zetaX-1.0e+00_alphaX-1.0_TvirminX-1.0e+00_aveTb028.94_Pop-1_256_300Mpc']

#creating list for storage
nf_R_list=[]
density_R_list=[]
Tb_R_list=[]

smoothing_length=np.linspace(0., 10., 11)

for i in smoothing_length:
    print "Current smoothing length=", i
    
    #creating boxes
    nf_Fcoll=files_in_func(nf_files_in, 0, sphere_radius=i)
    nf_box=files_in_func(nf_files_in, 1)
    density_Fcoll=files_in_func(density_files_in, 0, sphere_radius=i)
    density_box=files_in_func(density_files_in, 1)
    Tb_Fcoll=files_in_func(Tb_files_in, 0, sphere_radius=i)
    Tb_box=files_in_func(Tb_files_in, 1)
    
    #Finding R's
    nf_cov=Find_Covariance(nf_Fcoll, nf_box)
    nf_R=Find_R(nf_cov)
    nf_R_list.append(nf_R)
    
    density_cov=Find_Covariance(density_Fcoll, density_box)
    density_R=Find_R(density_cov)
    density_R_list.append(density_R)
    
    Tb_cov=Find_Covariance(Tb_Fcoll, Tb_box)
    Tb_R=Find_R(Tb_cov)
    Tb_R_list.append(Tb_R)

plt.plot(smoothing_length, np.abs(nf_R_list))
plt.plot(smoothing_length, np.abs(density_R_list))
plt.plot(smoothing_length, np.abs(Tb_R_list))
plt.xlabel('Smoothing length')
plt.ylabel('Correlation')
plt.title('Correlation vs smoothing length with Filter')
plt.legend(['Neutral Fraction', 'Density', 'Brightness Temperature'], loc='upper right')

plt.show()
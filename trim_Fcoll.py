import numpy as np
import sys

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

def reshape_data(data, dim):
    """
    Reshapes the data depending on its dimensions
    """
    data.shape = (dim, dim, dim + 2)
    data = data[:,:,:-2]  # slice so same size as other boxes

    data = data.copy(order='C')

    return data


in_path = "C:/Users/Ronnie/Documents/21cmFAST-msci/Boxes/Fcoll_output_file_z007.50_64_75Mpc"
DIM = 64
out_path = "C:/Users/Ronnie/Documents/21cmFAST-msci/Boxes/Fcoll_output_file_CUBE_z007.50_64_75Mpc"


Fcoll = load_binary_data(in_path)
print Fcoll.flags
Fcoll = reshape_data(Fcoll, DIM)
print Fcoll.flags
write_binary_data(out_path, Fcoll)

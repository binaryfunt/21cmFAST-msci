USAGE = "run trim_Fcoll.py -i <Fcoll file in (box)>"

import numpy as np
import sys
import getopt

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


in_path = ""

try:
    opts, args = getopt.getopt(sys.argv[1:], "h:i:")
except getopt.GetoptError:
    print USAGE
    sys.exit(2)
# print opts, args
for opt, arg in opts:
    if opt in ("-h", "--h", "--help"):
        print USAGE
        sys.exit()
    elif opt in ("-i"):
        in_path = arg
if in_path == "":
    print "USAGE:", USAGE
    sys.exit()

DIM = int("" + in_path.split("_")[-2])
label = str("" + in_path.split("_")[-1])
if in_path[-3:] == "Mpc":
    datafile_split = in_path.split("_")
    out_path = "_".join(datafile_split[:-2]) + "_CUBE_" + "_".join(datafile_split[-2:])
else:
    out_path = in_path + "CUBE"

Fcoll = load_binary_data(in_path)
# print Fcoll.flags
Fcoll = reshape_data(Fcoll, DIM)
# print Fcoll.flags
write_binary_data(out_path, Fcoll)

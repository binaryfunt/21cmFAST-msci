import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys, getopt

USAGE = "USAGE: Fcoll_histogram.py -i <Fcoll data file> [--number=<number of bins>]"

file_in = ""
number = 50

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

try:
    opts, args = getopt.getopt(sys.argv[1:], "h:i:n:", ["number="])
except getopt.GetoptError:
    print USAGE
    sys.exit(2)
# print opts, args
for opt, arg in opts:
    if opt in ("-h", "--h", "--help"):
        print USAGE
        sys.exit()
    elif opt in ("-i"):
        file_in = arg
    elif opt in ("-n", "--number"):
        number = int(arg)
if file_in == "":
    print USAGE
    sys.exit()

print "Opening Fcoll file at", file_in
Fcoll = load_binary_data(file_in)
# Fcoll2 = 1./Fcoll
# Fcoll3 = Fcoll2[Fcoll2 < 1e308]

print "Number of bins:", number

fig = plt.figure()
ax = plt.subplot()

n, bins, patches = plt.hist(Fcoll, number)

plt.xlabel("Collapse fraction")
plt.ylabel("Frequency")
ax.set_yscale("log")
ax.set_xscale("log")

plt.show()

import numpy as np
import matplotlib.pyplot as plt
import sys, getopt

USAGE = "USAGE: Fcoll_histogram.py -i <Fcoll data file>"

file_in = ""

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
        print "Opening Fcoll file at", arg
        file_in = arg
        break
if not opts:
    print USAGE
    sys.exit()

Fcoll = load_binary_data(file_in)

fig = plt.figure()
ax = plt.subplot()

n, bins, patches = plt.hist(Fcoll, 50)

plt.xlabel("Collapse fraction")
plt.ylabel("Frequency")
ax.set_yscale("log")

plt.show()

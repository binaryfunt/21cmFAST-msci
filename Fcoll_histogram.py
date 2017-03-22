import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.font_manager as font_manager
import sys, getopt

USAGE = "USAGE: Fcoll_histogram.py -i <Fcoll data file> [--number=<number of bins>] [--loglog] [--logy] [--savefig]"


font_path = 'C:\Windows\Fonts\Roboto-Regular.ttf'
font_prop = font_manager.FontProperties(fname=font_path, size=11)
title_fontsize = 13
tick_fontsize = 11
def set_ax_font(ax):
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontproperties(font_prop)
        label.set_fontsize(tick_fontsize)
        # label.font_properties(font_prop)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
b = "#0066DD"
g = "#00BB00"
m = "#DD00FF"


file_in = ""
number = 50
log_log = False
log_y = False
savefig = False

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

def histogram(data, number):
    _freq, _bin_edges = np.histogram(data, bins=number)
    _bin_centres = (_bin_edges[:-1] + _bin_edges[1:])/2.
    return _freq, _bin_centres

def monomial(x, a, k):
    return a*x**k


try:
    opts, args = getopt.getopt(sys.argv[1:], "h:i:n:", ["number=", "loglog", "logy", "savefig"])
except getopt.GetoptError:
    print USAGE
    sys.exit(2)
print opts, args
for opt, arg in opts:
    if opt in ("-h", "--h", "--help"):
        print USAGE
        sys.exit()
    elif opt in ("-i"):
        file_in = arg
    elif opt in ("-n", "--number"):
        number = int(arg)
    elif opt in ("--loglog", "--log"):
        log_log = True
    elif opt in ("--logy"):
        log_y = True
    elif opt in ("--savefig"):
        savefig = True
if file_in == "":
    print USAGE
    sys.exit()

print "Opening Fcoll file at", file_in
Fcoll = load_binary_data(file_in)
# Fcoll2 = 1./Fcoll
# Fcoll3 = Fcoll2[Fcoll2 < 1e308]

print "Number of bins:", number

# n, bins, patches = plt.hist(Fcoll, number, histtype='step')
freq, bin_centres = histogram(Fcoll, number)

initial_guess = [1e7, -2]

popt, pcov = curve_fit(monomial, bin_centres[1:-1], freq[1:-1], p0=initial_guess)
# N.B. We ignore the first and last bins in the fitting

print "Fitted a =", popt[0]
print "Fitted k =", popt[1]
print 100. * float(freq[0]) / float(sum(freq)), "% in 1st bin"
print 100. * float(freq[-1]) / float(sum(freq)), "% in last bin"

fig = plt.figure()
ax = plt.gca()

plt.plot(bin_centres, freq, marker='o', mec=b, mfc="w", linestyle="none", label="Data")
plt.plot(bin_centres, monomial(bin_centres, *popt), 'r-', label="Fit")

plt.xlabel("Collapse fraction", fontproperties=font_prop)
plt.ylabel("Frequency", fontproperties=font_prop)
if log_log == True:
    ax.set_yscale("log")
    ax.set_xscale("log")
elif log_y == True:
    ax.set_yscale("log")
plt.title(r"21cmFAST ${\rm f_{coll}}$", fontproperties=font_prop, fontsize=title_fontsize, verticalalignment='bottom')
plt.legend(prop=font_prop, frameon=False)
set_ax_font(ax)
plt.xlim(xmax=1.1)

if savefig == False:
    plt.show()
else:
    plt.savefig("Fcoll_histogram.png", dpi=200, bbox_inches="tight")
    plt.close()

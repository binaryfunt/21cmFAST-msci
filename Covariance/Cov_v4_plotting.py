import numpy as np
import scipy as sp
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import itertools
from matplotlib import rcParams
from os.path import basename
import sys, getopt
import os

font_path = 'C:\Windows\Fonts\Roboto-Regular.ttf'
font_prop = font_manager.FontProperties(fname=font_path, size=11)
title_fontsize = 13
tick_fontsize = 11
def set_ax_font(ax):
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontproperties(font_prop)
        label.set_fontsize(tick_fontsize)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
b = "#0066DD"
g = "#00BB00"
m = "#DD00FF"


# MODE = "Gauss"
MODE = "Sphere"

if MODE == "Gauss":
    data = np.loadtxt("gaussian-plot-data")
elif MODE == "Sphere":
    data = np.loadtxt("spherical-plot-data")
else:
    raise Exception("Filter MODE not specified correctly")

smoothing_length = data[0]
density_R_list = data[1]
nf_R_list = data[2]
Tb_R_list = data[3]

plt.figure(figsize=(5.5,5))
ax = plt.subplot() # Defines ax variable by creating an empty plot
set_ax_font(ax)

plt.plot(smoothing_length, np.abs(density_R_list), "--", color=b, label='Density')
plt.plot(smoothing_length, np.abs(nf_R_list), "-", color=g, label='Neutral fraction')
plt.plot(smoothing_length, np.abs(Tb_R_list), "-", color="r", label='Brightness temperature')
plt.xlabel('Smoothing scale (pixels)', fontproperties=font_prop)
plt.ylabel('Pearson correlation coefficient', fontproperties=font_prop)
if MODE == "Gauss":
    plt.title(r'${\rm f_{coll}}$ correlation (Gaussian filter)', fontproperties=font_prop, size=title_fontsize, verticalalignment='bottom')
elif MODE == "Sphere":
    plt.title(r'${\rm f_{coll}}$ correlation (spherical filter)', fontproperties=font_prop, size=title_fontsize, verticalalignment='bottom')
plt.legend(loc='upper right', prop=font_prop, frameon=False)

plt.xlim((min(smoothing_length), max(smoothing_length)))
plt.ylim((0,1))

# plt.show()
if MODE == "Gauss":
    plt.savefig("C:/Users/Ronnie/OneDrive/Documents/MSci Project/Viva/PCC-gaussian.png", dpi=200)
elif MODE == "Sphere":
    plt.savefig("C:/Users/Ronnie/OneDrive/Documents/MSci Project/Viva/PCC-spherical.png", dpi=200)

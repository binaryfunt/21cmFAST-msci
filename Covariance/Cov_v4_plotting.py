import numpy as np
import scipy as sp
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from os.path import basename
import sys, getopt
import os

font_path = 'C:\Windows\Fonts\Roboto-Regular.ttf'
font_prop = font_manager.FontProperties(fname=font_path, size=14)

# data = np.loadtxt("spherical-plot-data")
data = np.loadtxt("gaussian-plot-data")

smoothing_length = data[0]
density_R_list = data[1]
nf_R_list = data[2]
Tb_R_list = data[3]

plt.figure(figsize=(6,5))
ax = plt.subplot() # Defines ax variable by creating an empty plot
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontproperties(font_prop)
    label.set_fontsize(13)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

plt.plot(smoothing_length, np.abs(density_R_list), label='Density')
plt.plot(smoothing_length, np.abs(nf_R_list), label='Neutral fraction')
plt.plot(smoothing_length, np.abs(Tb_R_list), label='Brightness temperature')
plt.xlabel('Smoothing scale (pixels)', fontproperties=font_prop)
plt.ylabel('Pearson correlation coefficient', fontproperties=font_prop)
plt.title(r'${\rm f_{coll}}$ correlation (Gaussian filter)', fontproperties=font_prop, size=16, verticalalignment='bottom')
plt.legend(loc='upper right', prop=font_prop, frameon=False)

# plt.show()
plt.savefig("C:/Users/Ronnie/OneDrive/Documents/Research Interfaces/Poster/PCC-gaussian.png", dpi=200)

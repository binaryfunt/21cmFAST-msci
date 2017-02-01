import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0, 1.95, 40)
y=np.loadtxt('data3.txt', skiprows=2)

plt.plot(x,y)
plt.xlabel('Smoothing length')
plt.ylabel('Correlation, R')
plt.title('Fcoll and Tb')

plt.show()

import numpy as np
import matplotlib.pyplot as plt
x=np.x=np.random.normal(size=25)
y=np.random.normal(size=25)

#plt.scatter(x,y)

cov=np.cov(x,y)

r=cov[0,1]/(cov[0,0]*cov[1,1])

print r

#plt.show()
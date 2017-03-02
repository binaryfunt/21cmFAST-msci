import numpy as np
import matplotlib.pyplot as plt
x=np.x=np.random.normal(size=25)
y=np.random.normal(size=25)
#print x,y

#plt.scatter(x,y)

cov=np.zeros((2,2))
print cov

def covariance(vec1,vec2=None):
    if vec2==None:
        vec2=vec1
    
    a=0.
    for i in range(len(vec1)):
        a+=vec1[i]*vec2[i]
    
    a=a/(len(vec1))
    
    b=vec1.mean()*vec2.mean()
    
    return a + b

print covariance(x,x)
print covariance(x,y)
print covariance(y,x)
print covariance(y,y)

#plt.show()
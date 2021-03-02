import numpy as np

X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1
for j in xrange(60000):
		I1 = 1/(1 + np.exp(-(np.dot(X, syn0))))
		I2 = 1/(1 + np.exp(-(np.dot(I1, syn1))))
		I2_delta = (y-I2)*(I2*(1-I2))
		I1_delta = I2_delta.dot(syn1.T)*(I1*(1-I1))
		syn1 += I1.T.dot(I2_delta)
		syn0 += X.T.dot(I1_delta)
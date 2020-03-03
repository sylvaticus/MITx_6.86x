# -*- coding: utf-8 -*-
"""
Unit 00 -  Brief Review of Vectors, Planes, and Optimization
"""

from sympy import *
x, p1, p2 = symbols('x p1 p2')
y = 1/(1+exp( - (p1*x + p2)))
dy_dp1 = diff(y,p1) 
print(dy_dp1)

import numpy as np
a = np.array([4,1])
b = np.array([2,3]) 
normC = np.dot(a,b)/np.linalg.norm(b)
c = (np.dot(a,b)/np.linalg.norm(b)**2) * b


from scipy.stats import norm
rv = norm(1, scale=sqrt(2.))
P = rv.cdf(2) - rv.cdf(0.5)


A = np.array([[1,2,3], [4, 5, 6], [1, 2, 1]])
B = np.array([[2, 1, 0], [1, 4, 4], [5, 6, 4]])

np.linalg.matrix_rank(A)
np.linalg.matrix_rank(B)

np.linalg.det(A) 


# ## Project 0 Setup, Numpy Exercises, Tutorial on Common Packages

# ### 4. Introduction to Numpy
 
# Neural network
 
# - Inputs: X = [x1, x2]
# - Outputs: z
# - Weigths: W = [w1,w2]
 
# $z = f(w1x1 + w2x2)$
 
# Where f is typically a non-linear function as logistic, sigmoid or hiperbolic tangent ("tanh").
 
# We use tanh.
 
import numpy as np
 
x = np.array([1,5]) 
 
np.zeros([4,5]) 

np.random.random([3,2])

y = x.reshape((2,1)) 
x
y

def myfunction(a):
   return a+1

myfunction(2)

x = np.array([[1,2,3],[4,5,6]]) 
x.shape
y = x.transpose() 

x = np.array([3,5])
z = np.array([[6,8,4],[4,6,12]])
np.matmul(x,z)
np.matmul(x.reshape((1,2)),z)
y = z + 1
z * y
np.tanh(x)

x = np.array([4,2,9,-6,5,11,3])
np.random
np.linalg.norm(x)
np.transpose(x)

a = np.array([1,2,3])
b = np.array([10,20,30])
np.shape(a)
np.dot(a,b)
np.matmul(a.T,b)

a = np.array([1,2,3]).reshape(3,1)
b = np.array([10,20,30]).reshape(3,1)
np.shape(a)
np.dot(a,b) # error
np.dot(a.T,b)
np.matmul(a.T,b)

def myfunction2(a):
   a+1
   
b = myfunction2(3)  
   





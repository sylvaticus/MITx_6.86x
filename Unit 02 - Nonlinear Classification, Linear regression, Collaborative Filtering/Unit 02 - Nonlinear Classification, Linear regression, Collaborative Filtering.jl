# Unit 02 - Nonlinear Classification, Linear regression, Collaborative Filtering

# Lecture 5

# Hinge loss computation

x = [1 0 1; 1 1 1; 1 1 -1; -1 1 1]
y = [2,2.7,-0.7, 2]
θ = [0,1,2]
z = y - (θ' * x')'

hinge(z) = (z >= (1 - eps()) ) ? 0 : (1 - z)
l = mean(map(z -> hinge(z), z))

squared_error(z) = z^2 / 2
l = mean(map(z -> squared_error(z), z))

# Lecture 6

# radial basis kernel
using LinearAlgebra
radial_kernel(x,xᵖ) = exp(-1/2 * norm(x-xᵖ)^2)

x = [1,0,0]

xᵖ = [0,1,0]

k = radial_kernel(x,xᵖ)

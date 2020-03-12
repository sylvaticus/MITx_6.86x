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

# Homework 3, 1. Collaborative Filtering, Kernels, Linear Regression

Y = [5 missing 7; missing 2 missing; 4 missing missing; missing 3 6]
λ = 1
u₀ = [6,0,3,6]
v₀ = [4,2,1]
(n,m) = size(Y)

X₀ = u₀ * v₀'
L₁ = sum(skipmissing((Y-X₀) .^ 2))/2
ϵ₁ = (λ/2) * (norm(u₀)^2 + norm(v₀)^2)

#using SymPy, LinearAlgebra
#u,v,Y,λ = symbols("u,v,Y,λ", real=true, positive=true)

J(u,v,Y,λ) =  sum(skipmissing((Y-(u * v')) .^ 2))/2 + (λ/2) * (norm(u)^2 + norm(v)^2)
#J2 =  sum(skipmissing((Y-(u * v')) .^ 2))/2 + (λ/2) * (norm(u)^2 + norm(v)^2)
#dJ_dU(u,v,Y,λ)  = ( - (Y-u * v') * v) + λ * u

#dJ_dU(u₀,v₀,Y,λ)

J([27/18,4/5,16/17,12/6],v₀,Y,λ)

# 3. Kernels

# 4. Kernels-II

# 5. Linear Regression and Regularization

x = [2,4,6]

x^2

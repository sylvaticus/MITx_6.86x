# Unit 4 _ Unsupervised learning

using LinearAlgebra
using Random
using Distributions
using Statistics

# Lecture 13

# 13.6. Similarity Measures-Cost functions

# Cost of cluster

cost(members,z) = sum([norm(member - z')^2 for member in  eachrow(members)])

c1 = cost([-1 2; -2 1; -1 0],[-1 1])
c2 = cost([2 1; 3 2],[2 2])

c = c1+c2

# Sterling number: number of partitions of a set of n elements in k sets:

sterling(n::BigInt,k::BigInt) = (1/factorial(k)) * sum((-1)^i * binomial(k,i)* (k-i)^n for i in 0:k)
sterling(n::Int64,k::Int64) = sterling(BigInt(n),BigInt(k))
sterling(3,2)
sterling(BigInt(40),BigInt(3))
sterling(100,3)

# 13.7 The K-Means Algorithm
https://github.com/sylvaticus/lmlj.jl/blob/master/src/clusters.jl#L65

# 14.3 - K-Medoids algorithm
https://github.com/sylvaticus/lmlj.jl/blob/master/src/clusters.jl#L133

# 15.9

x = [1/sqrt(π), 2]
μ = [0,2]
σ = sqrt(1/2π)

normal(x,μ,σ²) = (1/(2π*σ²)^(length(x)/2)) * exp(-1/(2σ²)*norm(x-μ)^2)

p = normal(x',μ',σ^2)
lp = log(p)

# 16.4

# Checking log(sum(x)) = sum(log(x))
x = [0.2,0.8,1,4.5,7]

log(sum(x))
sum(log.(x))

K = 2
X = [-1.2 -0.8; -1 -1.2; -0.8 -1; 1.2 0.8; 1 1.2; 0.8 1]
cIdx = [1,1,1,2,2,2]
X₁ = X[cIdx .== 1,:]
X₂ = X[cIdx .== 2,:]

μ₁ = mean(X₁ , dims=1)
μ₂ = mean(X₂ , dims=1)

# Checking log(sum(x * p(x))) = sum(log(x) * px)
x = [0.2,0.8,1,4.5,7]
px = [0.1,0.4,0.2,0.1,0.2]

log(sum(x .* px))
sum(log.(x) .* px)
sum(log.(x .* px))


# 16.5 The E-M Algorithm
https://github.com/sylvaticus/lmlj.jl/blob/master/src/clusters.jl#L222

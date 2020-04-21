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

"""
  kmean(X,K)

Compute K-Mean algorithm to identify K clusters of X using Euclidean distance

# Parameters:
* `X`: a (n x d) data to clusterise
* `K`: Number of cluster wonted

# Returns:
* A vector of size n of ids of the clusters associated to each point

# Notes:
* Some returned clusters could be empty

# Example:
```julia
julia> clIdx = kmean([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.6 38],2)
```
"""
function kmean(X,K)
    (n,d) = size(X)
    # Random choice of initial representative vectors (any point, not just in X!)
    minX = minimum(X,dims=1)
    maxX = maximum(X,dims=1)
    Z₀ = zeros(K,d)
    for i in 1:K
        for j in 1:d
            Z₀[i,j] = rand(Uniform(minX[j],maxX[j]))
        end
    end
    Z = Z₀
    cIdx_prev = zeros(Int64,n)

    # Looping
    while true
        # Determining the constituency of each cluster
        cIdx      = zeros(Int64,n)
        for (i,x) in enumerate(eachrow(X))
            cost = Inf
            for (j,z) in enumerate(eachrow(Z))
               if (norm(x-z)^2  < cost)
                   cost    =  norm(x-z)^2
                   cIdx[i] = j
               end
            end
        end

        # Checking termination condition: clusters didn't move any more
        if cIdx == cIdx_prev
            return cIdx
        else
            cIdx_prev = cIdx
        end

        # Determining the new representative by each cluster
        for (j,z) in enumerate(eachrow(Z))
            Cⱼ = X[cIdx .== j,:] # Selecting the constituency by boolean selection
            z = sum(Cⱼ,dims=1) ./ size(Cⱼ)[1]
        end
    end
end

clIdx = kmean([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.6 38],2)

# 14.3 - K-Medoids algorithm

"""Square Euclidean distance"""
square_euclidean(x,y) = norm(x-y)^2

"""Cosine distance"""
cos_distance(x,y) = dot(x,y)/(norm(x)*norm(y))


"""
  kmedoids(X,K;dist)

Compute K-Medoids algorithm to identify K clusters of X using distance definition `dist`

# Parameters:
* `X`: a (n x d) data to clusterise
* `K`: Number of cluster wonted
* `dist`: Function to employ as distance (must accept two vectors). Default to squared Euclidean.

# Returns:
* A vector of size n of ids of the clusters associated to each point

# Notes:
* Some returned clusters could be empty

# Example:
```julia
julia> clIdx = kmedoids([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.6 38],2,dist = (x,y) -> norm(x-y)^2)
```
"""
function kmedoids(X,K;dist=(x,y) -> norm(x-y)^2)
    (n,d) = size(X)
    # Random choice of initial representative vectors
    zIdx = shuffle(1:size(X)[1])[1:K]
    Z₀ = X[zIdx, :]
    Z = Z₀
    cIdx_prev = zeros(Int64,n)

    # Looping
    while true
        # Determining the constituency of each cluster
        cIdx      = zeros(Int64,n)
        for (i,x) in enumerate(eachrow(X))
            cost = Inf
            for (j,z) in enumerate(eachrow(Z))
               if (dist(x,z) < cost)
                   cost =  dist(x,z)
                   cIdx[i] = j
               end
            end
        end

        # Checking termination condition: clusters didn't move any more
        if cIdx == cIdx_prev
            return cIdx
        else
            cIdx_prev = cIdx
        end

        # Determining the new representative by each cluster (within the points member)
        for (j,z) in enumerate(eachrow(Z))
            Cⱼ = X[cIdx .== j,:] # Selecting the constituency by boolean selection
            nⱼ = size(Cⱼ)[1]     # Size of the cluster
            println(nⱼ)
            if nⱼ == 0 continue end # empty continuency. Let's not do anything. Stil in the next batch other representatives could move away and points could enter this cluster
            bestCost = Inf
            bestCIdx = 0
            for cIdx in 1:nⱼ      # candidate index
                 candidateCost = 0.0
                 for tIdx in 1:nⱼ # target index
                     candidateCost += dist(Cⱼ[cIdx,:],Cⱼ[tIdx,:])
                 end
                 if candidateCost < bestCost
                     bestCost = candidateCost
                     bestCIdx = cIdx
                 end
            end
            z = reshape(Cⱼ[bestCIdx,:],1,d)
        end
    end
end

clIdx = kmedoids([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.6 38],2,dist = (x,y) -> norm(x-y)^2)

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


""" PDF of a multidimensional normal with no covariance and shared variance across dimensions"""
normalFixedSd(x,μ,σ²) = (1/(2π*σ²)^(length(x)/2)) * exp(-1/(2σ²)*norm(x-μ)^2)

# 16.5 The E-M Algorithm
"""
  em(X,K;p₀,μ₀,σ²₀,tol)

Compute Expectation-Maximisation algorithm to identify K clusters of X data assuming a Gaussian Mixture probabilistic Model.

# Parameters:
* `X`: a (n x d) data to clusterise
* `K`: Number of cluster wanted
* `p₀`: Initial probabilities of the categorical distribution (K x 1) [default: `nothing`]
* `μ₀`: Initial means (K x d) of the Gaussian [default: `nothing`]
* `σ²₀`: Initial variance of the gaussian (K x 1). We assume here that the gaussian has the same variance across all the dimensions [default: `nothing`]
* `tol`: Initial tolerance to stop the algorithm [default: 0.0001]

# Returns:
* A matrix of size n x K of the probabilities of each point i to belong to cluster j

# Notes:
* Some returned clusters could be empty

# Example:
```julia
julia> clIdx = em([1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.6 38],2)
```
"""
function em(X,K;p₀=nothing,μ₀=nothing,σ²₀=nothing,tol=0.0001)
# debug:
X = [1 10.5;1.5 10.8; 1.8 8; 1.7 15; 3.2 40; 3.6 32; 3.6 38]
K = 3
p₀=nothing; μ₀=nothing; σ²₀=nothing; tol=0.0001
(N,D) = size(X)
# Random choice of initial representative vectors (any point, not just in X!)
minX = minimum(X,dims=1)
maxX = maximum(X,dims=1)
varX = mean(var(X,dims=1))/K^2

# Initialisation of the parameters if not provided
p = isnothing(p₀) ? fill(1/K,K) : p₀
if !isnothing(μ₀)
    μ = μ₀
else
    μ = zeros(Float64,K,D)
    tempμ = collect(range(minX, stop=maxX, length=K))
    for k in K
        for d in D
            μ[k,d] = tempμ[k][d]
        end
    end
end
σ² = isnothing(σ²₀) ? fill(varX,K) : σ²₀

pⱼₓ = zeros(Float64,N,K)

for n in 1:N
    println([p[j] for j in 1:K])
    px = sum([p[j]*normalFixedSd(x[n,:],μ[j,:],σ²[j]) for j in 1:K])
    for k in 1:K
    #    pⱼₓ[n,k] = p[k]*normalFixedSd(x[n,:],μ[k,:],σ²[k])/px
    end
end

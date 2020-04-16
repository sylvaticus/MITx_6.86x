# Unit 4 _ Unsupervised learning

using LinearAlgebra
using Random

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
"""Square Euclidean distance"""
square_euclidean(x,y) = norm(x-y)^2

"""
  km_euclidean(X,K)

Compute K-Mean algorithm to identify K clusters of X using Euclidean distance

# Parameters:
* X: a (n x d) data to clusterise
* K: Number of cluster wonted

# Returns:
* A vector of size n of ids of the cluster associated to each point

# Example:
```julia
julia> cIdx = km_euclidean([1 1.5;1.5 1.8; 1.8 0.8; 1.7 1.5; 3.2 4; 3.6 3.2; 3.6 3.8],2)
```
"""
function km_euclidean(X,K,dist=square_euclidean)
    (n,d) = size(X)
    # Random choice of initial representative vectors
    zIdx = shuffle(1:size(X)[1])[1:K]
    Z₀ = X[zIdx, :]
    println(Z₀)
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

        # Determining the new representative by each cluster
        for (j,z) in enumerate(eachrow(Z))
            Cⱼ = X[cIdx .== j,:] # Selecting the constituency by boolean selection
            z = sum(Cⱼ,dims=1) ./ size(Cⱼ)[1]
        end
    end
end

km_euclidean([1 1.5;1.5 1.8; 1.8 0.8; 1.7 1.5; 3.2 4; 3.6 3.2; 3.6 3.8],2)

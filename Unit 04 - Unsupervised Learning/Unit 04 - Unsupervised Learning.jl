# Unit 4 _ Unsupervised learning

using LinearAlgebra

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

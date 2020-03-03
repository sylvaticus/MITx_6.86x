# Unit 01 - Linear Classifiers and Generalizations

function perceptron(set, T=1000)
    n = length(set); d = length(set[1][1])
    θ = zeros(d); θ₀ = 0;
    for t in 1:T
        ϵ = 0
        for i in 1:n
            x = set[i][1]; y = set[i][2]
            if (y*(θ' * x + θ₀) <= 0)
                θ  = θ + y * x
                θ₀ = θ₀ + y
                ϵ += 1
            end
        end
        if (ϵ == 0)
            return (θ,θ₀,ϵ,t)
        end
    end
    return (θ,θ₀,ϵ,T)
end

T = 5000
set = [([7,8],-1),([4,2],-1),([2,7],1),([3,-2],1),([-3,-2],1)]
(θ,θ₀,ϵ,t) = perceptron(set,T)

using Plots, LinearAlgebra

θₙ = θ/norm(θ)
θ₀ₙ = θ₀/norm(θ) *  θₙ
x⁺ = vcat([([tuple[1][1] tuple[1][2]])  for tuple in set if tuple[2] == 1]...)
x⁻ = vcat([([tuple[1][1] tuple[1][2]])  for tuple in set if tuple[2] == -1]...)

scatter(x⁺[:,1],x⁺[:,2], label="+1")
scatter!(x⁻[:,1],x⁻[:,2], label="-1")
#plot!([0,θ[1]],[0,θ[2]])
plot!([-θ₀ₙ[1],θₙ[1]-θ₀ₙ[1]],[-θ₀ₙ[2],θₙ[2]-θ₀ₙ[2]], arrow=1., label = "theta" )


set = [([3,3],1),([4,2],1),([2,7],-1),([3,-2],-1)]

function perceptronOrigin(set, T=30)
    println("*** New Lookup ***")
    n  = length(set); d = length(set[1][1])
    θ  = zeros(d)
    ϵt = 0
    for t in 1:T
        ϵ = 0
        for i in 1:n
            x = set[i][1]; y = set[i][2]
            error = false
            θpre = θ
            if (y*(θ' * x) <= 0)
                θ  = θ + y * x
                ϵ += 1; error = true; ϵt += 1
            end
            println("θpre: $θpre x: $x y: $y error: $error θpost=$θ")
        end
        if (ϵ == 0)
            return (θ,ϵ,t, ϵt)
        end
    end
    return (θ,ϵ,T,ϵt)
end

set = [([-1,-1],1),([1,0],-1),([-1,1.5],1)]
perceptronOrigin(set)

set = [([1,0],-1),([-1,1.5],1),([-1,-1],1)]
perceptronOrigin(set)

set = [([-1,-1],1),([1,0],-1),([-1,10],1)]
perceptronOrigin(set)

set = [([1,0],-1),([-1,10],1),([-1,-1],1)]
perceptronOrigin(set)

function perceptronFull(set; T=30, θ = zeros(length(set[1])), θ₀ = 0)
    println("*** New Classifier Lookup ***")
    println("Data: $set")
    println("Parameters: T = $T, θ=$θ, θ₀=$θ₀")
    n = length(set); d = length(set[1][1])
    for t in 1:T
        ϵ = 0
        for i in 1:n
            x = set[i][1]; y = set[i][2]
            error = false
            θpre = θ
            θ₀pre = θ₀
            if (y*(θ' * x + θ₀) <= 0)
                θ  = θ + y * x
                θ₀ = θ₀ + y
                ϵ += 1
                error = true
            end
            println("θpre: $θpre θ₀pre: $θ₀pre x: $x y: $y error: $error θpost=$θ θ₀post=$θ₀")
        end
        if (ϵ == 0)
            return (θ,θ₀,ϵ,T)
        end
    end
    return (θ,θ₀,ϵ,T)
end

set = [([-4,2],1),([-2,1],1),([-1,-1],-1),([2,2],-1),([1,-2],-1)]
(θ,θ₀,ϵ,T) = perceptronFull(set)

## Homework 1

### Tab 2

scatter([-1,1],[1,-1],label="positive")
scatter!([1,2],[1,2],label="negative")



## Homework 2

### Tab 1

λ = 0.5
x = [1,0]
y = 1

θ =
loss(θ,y=1,x=[1,0],λ=0.5) = max(0,1-y*(θ' * x)) + λ/2 * norm(θ)^2

loss([0,0])
loss([1,0])
loss([0.5,0])
loss([-1,0])
loss([1,-1])

max(0,1-1*([1,0]' * [1,0]))
max(0,1-1*([0.5,0]' * [1,0]))

norm()

### Tab 3

cos(0)
cos(π)
cos(2*π)
cos(3*π)
cos(4*π)

function setFactory(d,labels=ones(d))
    out = Tuple{Array{Int64,1},eltype(labels)}[] # or just out=[]
    #out = []
    for i in 1:d
      x = zeros(d)
      x[i] = cos(i*π)
      push!(out,(x,labels[i]))
    end
    return out
end

set = setFactory(20)
using Random
shuffle!(set)
(θ,ϵ,t,ϵt) = perceptronOrigin(set,400)

randLabel = rand(20)
set = setFactory(20,randLabel)
(θ,ϵ,t,ϵt) = perceptronOrigin(set,400)
shuffle!(set)
(θ,ϵ,t,ϵt) = perceptronOrigin(set,400)

randLabel = rand(3)
set = setFactory(3,randLabel)
(θ,ϵ,t,ϵt) = perceptronOrigin(set,400)

set = setFactory(3)
(θ,ϵ,t,ϵt) = perceptronOrigin(set,400)

set=setFactory(2)
(θ,ϵ,t,ϵt) = perceptronOrigin(set,400)

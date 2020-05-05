# Unit 05 - Reinforcement Learning

using LinearAlgebra
using Random
using Distributions
using Statistics

# Lesson 17

# Lesson 17.5
R0 = -10+0.5*(-1-10)+0.5^2 *1

R = -10+0.5*(-1+-10)+0.5^2+1
R2 = (-1+-10)+0.5*(+1-10)

R3 = (-1-10)+0.5*(+1-10)

R4 = -10+0.5*(-1-10)+0.5^2*1

-10-1+0.5*(-10+1) # correct

# 17.7

S = 5
γ = 0.5
T1 = [2/3 1/3 0   0   0  ;
      1/2 1/2 0   0   0  ;]
T2 = [0   2/3 1/3 0   0  ;
      1/4 1/2 1/4 0   0  ;
      1/3 2/3 0   0   0  ;]
T3 = [0   0   2/3 1/3 0  ;
      0   1/4 1/2 1/4 0  ;
      0   1/3 2/3 0   0  ;]
T4 = [0   0   0   2/3 1/3;
      0   0   1/4 1/2 1/4;
      0   0   1/3 2/3 0  ;]
T5 = [0   0   0   1/2 1/2;
      0   0   0   1/3 2/3;]


T1 = [2/3 1/3 0   0   0  ;
      1/2 1/2 0   0   0  ;
      1/2 1/2 0   0   0  ;]
T2 = [0   2/3 1/3 0   0  ;
      1/4 1/2 1/4 0   0  ;
      1/3 2/3 0   0   0  ;]
T3 = [0   0   2/3 1/3 0  ;
      0   1/4 1/2 1/4 0  ;
      0   1/3 2/3 0   0  ;]
T4 = [0   0   0   2/3 1/3;
      0   0   1/4 1/2 1/4;
      0   0   1/3 2/3 0  ;]
T5 = [0   0   0   1/2 1/2;
      0   0   0   1/2 1/2;
      0   0   0   1/3 2/3;]

T = [T1,T2,T3,T4,T5] # array of (nA x nS) transactions

R =[0,0,0,0,1] # Rewards

V = zeros(S)
#V = rand(5) # same results
K = 100
for k in 1:K
    Vlag = copy(V)
    rV = R + γ * Vlag
    for s in 1:S
        V[s] = maximum(T[s] * rV)
    end
end

V100 = copy(V)

# Original non vectorized version..
V = zeros(S)
K = 10
for k in 1:K
    Vlag = copy(V)
    for s in 1:S
        V[s] =  - Inf
        nA = size(T[s])[1] # number of actions from this state
        for a in 1:nA
            v = 0
            for sTo in 1:S
                v  +=  T[s][a,sTo]*(R[sTo] + γ*Vlag[sTo])
            end
            if v > V[s]
                V[s] = v
            end
        end
    end
end

# Lesson 18

# 18.3

S = 2
A = 1
γ = 0.5
α = 0.75
# This is a matrix of S states rows and A cols actions.
# There is only one action in the problem, so in this specific case
# Q is just a column vector
Q = zeros(S,A)
# Dimensions: id_stateFrom, idAction, id_stateTo, reward collected
# Note that in the stated problem there is only one action per departing state,
# so idAction is not really needed
data = [1 1 1 1;
        1 1 2 -1;
        2 1 1 1]
for d in eachrow(data)
    # Updating the Q-value of the combination (sFrom,a) observed in the sample
    # maximum(Q[d[3],:]) is the maximum Q of the sTo state given in the sample
    # In this case with just a single action is the maximum of a single value.
    # Note that we are dealing with live **averages**. All previous valus are
    # already averages, not the individual previous sample values.
    Q[d[1],d[2]] = Q[d[1],d[2]] - α * (Q[d[1],d[2]] - d[4] - γ * maximum(Q[d[3],:])  )
end

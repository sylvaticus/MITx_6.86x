 # Unit 03 - Neural networks

## Lecture 9

### 9.2

using LinearAlgebra, StatsPlots

X = [-1 -1; 1 -1; -1 1; 1 1]
Y = [1,-1,-1,1]

Wa = [0 0 0; 0 0 0]
Wb = [1 2 2; 1 -2 -2]
Wc = [1 -2 -2; 1 2 2]

f(z) = 2z-3

(n,d) = size(X)

Za = X * Wa[:,2:3]' .+ Wa[:,1]'
Zb = X * Wb[:,2:3]' .+ Wb[:,1]'
Zc = X * Wc[:,2:3]' .+ Wc[:,1]'

Fa = f.(Za)
Fb = f.(Zb)
Fc = f.(Zc)

colors = [y == 1 ? "blue" : "red" for y in Y]
scatter(Fb[:,1],Fb[:,2], colour=colors, label="", title="Out b")
scatter(Fc[:,1],Fc[:,2], colour=colors, label="", title="Out c")

W = [1 1 -1; 1 -1 1]

fa(z) = 5z-2
fb(z) = max(0,z)
fc(z) = tanh(z)
fe(z) = z

Z = X * W[:,2:3]' .+ W[:,1]'
Fa = fa.(Z)
Fb = fb.(Z)
Fc = fc.(Z)
Fe = fe.(Z)

scatter(Fa[:,1],Fa[:,2], colour=colors, label="", title="Out a")
scatter(Fb[:,1],Fb[:,2], colour=colors, label="", title="Out b")
scatter(Fc[:,1],Fc[:,2], colour=colors, label="", title="Out c")
scatter(Fe[:,1],Fe[:,2], colour=colors, label="", title="Out e")

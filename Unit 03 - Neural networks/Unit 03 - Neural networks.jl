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


## Lecture 10 Recurrent Neural Network

### 10.5

a = tanh(6)

b = tanh(2)

## Homework 4

### 4.1. Neural Networks

W = [1 0; 0 1; -1 0; 0 -1]
W₀ = [-1,-1,-1,-1]

V = [1 1 1 1; -1 -1 -1 -1]
V₀ = [0,2]

softmax(x,β=1) = exp.(β .* x) /sum(exp.(β .* x))

x = [3 14]
z = x * W' .+ W₀'

fz = max.(0,z)
u = fz * V' .+ V₀'
fu = max.(0,u)
o = softmax(fu)

# Output of Neural Network
u = [1 1]
o = softmax(max.(0,u))
u = [0 2]
o = softmax(max.(0,u))
u = [3 -1]
o = softmax(max.(0,u))

### 4.2. LSTM

f(h,x)   = 1/(1+exp(-(-100)))
i(h,x)   = 1/(1+exp(-(100*x+100)))
out(h,x) = 1/(1+exp(-(100*x)))
c(h,c,x) = f(h,x)*c+i(h,x)*tanh(-100*h+50*x)
h(h,c,x) = out(h,x)*tanh(c)

h₀ = 0
c₀ = 0
x  = [0,0,1,1,1,0]

function LSTM(h₀, c₀, X)
    hs = Float64[h₀]
    cs = Float64[c₀]
    for x in X
       ct = c(hs[end],cs[end],x)
       ht = round(h(hs[end],ct,x))
       push!(cs,ct)
       push!(hs,ht)
    end
    return (hs,cs)
end

(ho,co) = LSTM(h₀, c₀,x)
(ho,co) = LSTM(h₀, c₀,[1,1,0,1,1])
(ho,co) = LSTM(h₀, c₀,[1,1,1,1,1])
(ho,co) = LSTM(h₀, c₀,[1,1,0,0,1])
(ho,co) = LSTM(h₀, c₀,[1,0,1,0,1])

### 4.3. Backpropagation

f(x) = exp(-x)/(1+exp(-x))^2

f(2)
f(3)
f(4)
f(5)
f(2.5)

sigmoid(x)  = 1/(1+exp(-x))
dsigmoid(x) = exp(-x)/(1+exp(-x))^2

x  = 3
w1 = 0.01
w2 = -5
b  = -1
t  = 1

z1 = x*w1
a1 = max(0,z1)
z2 = w2*a1+b
y  = sigmoid(z2)
C = (1/2)*(y-t)^2

dw1 = x * w2 * dsigmoid(z2) * (y-t)
dw2 = a1 * dsigmoid(z2) * (y-t)
db  = 1 * dsigmoid(z2) * (y-t)

##  Lecture 12. Convolutional Neural Networks
### 12.2. Convolutional Neural Networks

fSupp = [1,2,3]
gSupp = [2,1]
f(x) = fSupp[x+1]
g(x) = gSupp[x+1]

function conv(f,g,n)
    for m in 0:length(fSupp)


f:        1 2 3
h(0) 2| 1 2
h(1) 5|   1 2
h(2) 8|     1 2
h(3) 3|       1 2


ReLU(x) = max(0,x)
x = [1 1 2 1 1;
     3 1 4 1 1;
     1 3 1 2 2;
     1 2 1 1 1;
     1 1 2 1 1]
w = [ 1 -2  0;
      1  0  1;
     -1  1  0]
(xr,xc) = size(x)
(wr,wc) = size(w)
z = [sum(x[r:r+wr-1,c:c+wc-1] .* w) for c in 1:xc-wc+1 for r in 1:xr-wr+1] # Julia is column mayor
u = ReLU.(z)
final = reshape(u, 1:xr-wr+1, 1:xc-wc+1)

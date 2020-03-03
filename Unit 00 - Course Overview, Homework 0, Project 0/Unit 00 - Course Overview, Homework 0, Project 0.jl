# Unit 00 -  Brief Review of Vectors, Planes, and Optimization

using LinearAlgebra, Distributions

## Homework 0

### 4. Points and Vectors

x = [0.4, 0.3]
y = [-0.15, 0.2]

nx = norm(x)
ny = norm(y)

sqrt(0.4^2+0.3^2)
sqrt((-0.15)^2+0.2^2)

dotxy = dot(x,y)
0.4*-0.15+0.3*0.2

α = acos(0)

3.14/2

using LinearAlgebra
a = [4,1]
b = [2,3]
normC = dot(a,b)/norm(b)
c = (dot(a,b)/norm(b)^2) * b

### 7. Univariate Gaussians

#### Probability

# Direct approach
X = Normal(1,sqrt(2))
P = cdf(X,2) - cdf(X,0.5)

# Using Standard Normals
Z = Normal(0,1)
ZlBound = (0.5-1)/sqrt(2)
ZuBound = (2-1)/sqrt(2)
P = cdf(Z,ZuBound) - cdf(Z,ZlBound)

### 8. (Optional Ungraded Warmup) 1D Optimization via Calculus

using SymPy,Plots

x = symbols("x")

f = (1/3)* x^3 - x^2 - 3*x + 10
df = diff(f,x)
critPoints = solve(df,x)
df2 = diff(df,x)

df2_x1 = subs(df2, x => critPoints[1])
df2_x2 = subs(df2, x => critPoints[2])

f_x1 = subs(f, x => critPoints[1])
f_x2 = subs(f, x => critPoints[2])
f_m4 = subs(f, x => -4)
f_4 = subs(f, x => 4)
# or, more simply...
f(critPoints[1])
f(critPoints[2])
f(-4)
f(4)

plot(f,-4,4)
plot(-exp(-x))
plot(x^0.7)

### 9. Gradients and Optimization

surface((x,y) -> x^2+y^2)

using Plots
x=-10:0.1:10
y=-10:0.1:10
f2(x,y) = x^2+y^2
plot(x,y,f2,st=:surface,camera=(45,45))

### 10. Matrices and Vectors

A = [1 2 3; 4 5 6; 1 2 1]
B = [2 1 0; 1 4 4; 5 6 4]

rank(A)
rank(B)

### 12. Linear Independence, Subspaces and Dimension

A = [1 3; 2 6]

B = [1 2; 2 1]

C = [1 1 0; 0 1 1; 0 0 1]

D = [2 -1 -1; -1 2 -1; -1 -1 2]

rank(A)

rank(B)

rank(C)

rank(D)

### 13. Determinant

A = [1 2 3; 4 5 6; 1 2 1]

det(A)

det(A')

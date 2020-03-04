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

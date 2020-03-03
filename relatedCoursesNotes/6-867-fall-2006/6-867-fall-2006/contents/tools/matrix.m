% Written by Jason Rennie <jrennie@ai.mit.edu> for 6.891, Sept. 2000

% Use the percent sign to write a comment
% THIS IS A SAMPLE COMMENT

% In order to get help from matlab, type 'help' or 'helpwin'
help
helpwin

% To get help on a specific function, type 'help function'
help sum

% Use these help commands to get some details on Matlab syntax
help paren
help punct

% Create a matrix with square brackets; put spaces (or commas) between elements
x = [1 2 3 4 5]
% Use semi-colon to not print result of command
y = [1.2,1.7,2.9,4.5,5.6];

% Create a vector of equally spaced values with the colon operator
x = 1:5
% Specify difference between values by splicing the difference between
% the start and end points
z = 1:0.5:5

% Access elements with parenthesis
x(2)
y(4)
y(5)

% Create a new row with a semi-colon
z = [1 2 3; 4 5 6]

% To perform matrix operations, use regular operators (e.g. +, -, *, /)
% To perform vector/array operations, use dot operators (e.g. `.*', `./')
z = x.*y

% To square the elements of an array, use the dot-power operator
z = x.^2

% Transpose a matrix with a single-quote
b = x*y'

% Use the 'size' function to get the width and height of a matrix
[n,m] = size(x'*y)

% Obtain the length of a vector, use the function 'length'
l = length(x)

% Invert a matrix with the command 'inv'
c = [1 2;2 1]
d = inv(c)

% Use the pseudo-inverse command, 'pinv', to invert singular matrices
c = [1 2;1 2]
d = pinv(c)


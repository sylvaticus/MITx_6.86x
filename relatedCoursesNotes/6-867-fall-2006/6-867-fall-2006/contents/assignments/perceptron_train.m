function [theta, k] = perceptron_train(X,y)
% [theta, k] = perceptron_train(X,y)
%   train a Perceptron classifier on given data
%
num_correct = 0;

[n,d] = size(X);

curr_index = 1;
theta = zeros(d,1);
k = 0;

is_first_iter = 1;

while (num_correct < n)
  xt = X(curr_index,:)';
  yt = y(curr_index);
  
  a = sign(yt*(theta'*xt));
  
  if (is_first_iter==1 | a < 0)
    num_correct = 0;
    theta = theta + (yt*xt);
    k = k+1;
    is_first_iter = 0;
  else
    num_correct = num_correct + 1;
  end
  
  curr_index = 1 + curr_index;
  if (curr_index > n)
    curr_index = 1;
  end
  
end

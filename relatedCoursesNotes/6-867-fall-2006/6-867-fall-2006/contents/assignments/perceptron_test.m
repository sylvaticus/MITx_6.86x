function test_err = perceptron_test(theta, X_test, y_test)
% test_err = perceptron_test(theta, X_test, y_test)
%   test a Perceptron classifier on given data and theta
%

[m, d] = size(X_test);
y_pred = sign(X_test*theta);
test_err = sum(abs(sign(y_test - y_pred)))/m;
end

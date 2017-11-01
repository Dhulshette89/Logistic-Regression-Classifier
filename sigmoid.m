%%% By the definition of the sigmoid function of logistic regression%%%

function g = sigmoid(z)
  g = zeros(size(z));
g = 1./(1 + e.^(-z));

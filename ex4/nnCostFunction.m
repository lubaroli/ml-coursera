function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Forward propagation

% Add bias unit to input features
X = [ones(m, 1) X];
% Compute L2 activation for all m samples (one column vector per sample)
z2 = Theta1*X';
a2 = sigmoid(z2);
% Compute L3 activation for all m samples (one column vector per sample)
z3 = Theta2*[ones(1, m); a2];
a3 = sigmoid(z3);
% Computes the prediction p and its confidence prob
[prob, p] = max(a3);

% Vectorize output y
yVec = zeros(num_labels, m);
for i = 1:m
	yVec(y(i), i) = 1;
end

J = (-1/m)*sum(sum(yVec.*log(a3) - (yVec - 1).*log(1 - a3)))...
	+ (lambda/(2*m))*(sum(nn_params.^2)...
	- sum(Theta1(:, 1).^2) - sum(Theta2(:, 1).^2));


% Back propagation

% To compute delta3 subtract the vectorized output of sample m yVec(:, m) from 
% the activation units of layer 3 for sample m a3(:, m)
delta3 = a3 - yVec;
% Use delta3 to compute delta2 for all m
delta2 = Theta2'*delta3 .* sigmoidGradient([ones(1, m); z2]);

% Sum the contributions in the Delta matrices
Delta1 = delta2(2:end, :)*X;
Delta2 = delta3*[ones(1, m); a2]';

% Compute gradients
Theta1_grad = (Delta1 + lambda*[zeros(size(Theta1, 1), 1), Theta1(:, 2:end)])/m;
Theta2_grad = (Delta2 + lambda*[zeros(size(Theta2, 1), 1), Theta2(:, 2:end)])/m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

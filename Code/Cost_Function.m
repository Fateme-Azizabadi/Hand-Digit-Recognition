function [J, grad] = Cost_Function(nn_params, ...
    input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
Y = (y==1:10);
Y = double(Y');
m = size(X, 1);

%forward pass
W1 = nn_params(1:(input_layer_size+1)*hidden_layer_size);
W2 = nn_params((input_layer_size+1)*hidden_layer_size+1:end);

W1 = reshape(W1, [hidden_layer_size, 1+input_layer_size]);
W2 = reshape(W2, [num_labels, 1+hidden_layer_size]);

W1_grad = zeros(size(W1));
W2_grad = zeros(size(W2));


X1 = [ones(size(X,1),1) X];
X1 = X1';
z1 = W1*X1;
y1 = sigmoid(z1);
X2=[ones(1, size(y1,2)); y1];
z2 = W2*X2;
y2 = sigmoid(z2);

% calculating cost function value
J = 1/m*sum(-Y.*log(y2)-(1-Y).*log(1-y2), 'all')+lambda/(2*m)*...
    (sum(W2.^2,'all')+sum(W1.^2,'all')-sum(W2(:,1).^2,'all')-sum(W1(:,1).^2,'all'));

% backward pass
Delta1 = 0;
Delta2 = 0;
for t = 1 : m
    a1 = X1(:, t);
    a2 = X2(:, t);
    a3 = y2(:, t);
    
    delta3 = a3 - Y(:, t);
    delta2 = (W2'*delta3).*sigmoid_Gradient([1; W1*a1]);
    delta2 = delta2(2:end);
    
    Delta2 = Delta2 + delta3*a2';
    Delta1 = Delta1 + delta2*a1';
end
Delta1 = Delta1/m;
Delta2 = Delta2/m;

W2_grad(:,2:end) = Delta2(:,2:end) +(lambda/m)*W2(:,2:end) ;
W2_grad(:,1) = Delta2(:,1);

W1_grad(:,2:end) = Delta1(:,2:end)+(lambda/m)*W1(:,2:end) ;
W1_grad(:,1) = Delta1(:,1);

grad = [W1_grad(:) ; W2_grad(:)];
end


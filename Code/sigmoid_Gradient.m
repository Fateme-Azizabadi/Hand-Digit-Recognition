function g = sigmoid_Gradient(z)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
g = exp(-z)./(1.0+exp(-z)).^2;
end


%%
clc
close all
clear variables

load('data.mat')
%%
%1.1
% Load dataset 
% plot randomely selected 100 images of the dataset
indices = randperm(size(X, 1), 100);
images = X(indices, :);
images = reshape(images, [100, 20, 20]);
images = permute(images, [2, 3, 1]);
montage(images)
%% 
%1.2
% Spliting dataset
[X_train, y_train, X_test, y_test] = train_test_split(X, y);
%%
%1.3
% Initialization
e = 0.12;
W12 = rand(25, 401)*2*e-e;
W23 = rand(10, 26)*2*e-e;
%%
% 1.6
input_layer_size = 400;
hidden_layer_size = 25;
num_labels = 10;
lambda=1;

initial_nn_params = [W12(:); W23(:)];

costFunction = @(p) Cost_Function(p, ...
input_layer_size, ...
hidden_layer_size, ...
num_labels, X_train, y_train, lambda);

options = optimset('MaxIter', 100);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% reshaping weigths
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
num_labels, (hidden_layer_size + 1));
%% 
%1.7
test_num = 25;
t = randperm(2000, test_num);
test = X_test(t, :);
test_label = y_test(t, :);

a1 = [ones(size(test,1),1) test];
a1 = a1';
z1 = Theta1*a1;
y1 = sigmoid(z1);
a2=[ones(1, size(y1,2)); y1];
z2 = Theta2*a2;
y2 = sigmoid(z2);

[~, predicted] = max(y2, [], 1);

test_images = reshape(test, [test_num, 20, 20]);
test_label(test_label==10) = 0;
predicted(predicted==10) = 0;

figure
for i = 1 : 25
    subplot(5, 5, i)
    imshow(squeeze(test_images(i, : , :))); 
    title(strcat('True Label:', num2str(test_label(i)), ', Predicted Label:', num2str(predicted(i))))
end
%% 
%1.8

mid_layer = reshape(Theta1(:, 2:end), [25, 20, 20]);

figure
for i = 1 : 25
    subplot(5, 5, i)
    imshow(squeeze(mid_layer(i, : , :)), []); 
    title(strcat('neuron ', num2str(i)))
end


% computing accuracy
a1 = [ones(size(X_test,1),1) X_test];
a1 = a1';
z1 = Theta1*a1;
y1 = sigmoid(z1);
a2=[ones(1, size(y1,2)); y1];
z2 = Theta2*a2;
y2 = sigmoid(z2);
[~, predicted] = max(y2, [], 1);

accuracy = sum(y_test(:)==predicted(:)) / length(y_test) * 100;

fprintf('Accuracy for 100 epochs is =%f\n', accuracy)
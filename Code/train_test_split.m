function [X_train, y_train, X_test, y_test] = train_test_split(X, y)

X_train = zeros(3000, 400);
y_train = zeros(3000, 1);
X_test = zeros(2000, 400);
y_test = zeros(2000, 1);

for i = 1 : 10
    X_train((i-1)*300+1:i*300, :) = X((i-1)*500+1:(i-1)*500+300, :);
    y_train((i-1)*300+1:i*300) = y((i-1)*500+1:(i-1)*500+300);
    X_test((i-1)*200+1:i*200, :) = X((i-1)*500+301:i*500, :);
    y_test((i-1)*200+1:i*200, :) = y((i-1)*500+301:i*500, :);
end
    
end


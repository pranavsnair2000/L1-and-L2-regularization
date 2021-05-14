
%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the X values and the third column
%  contains the label (y).
%  similarly X_test and y_test for test
data = load('data.txt');
[m,n]=size(data);
P=0.8;
idx=randperm(m);
train=data(idx(1:round(P*m)),:);
test=data(idx(round(P*m)+1:end),:);

X = train(:, [1, 2]); y = train(:, 3);
X_test = test(:, [1, 2]); y_test = test(:, 3);


plotData(X, y);
%pause;

% Put some labels
hold on;

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
title(sprintf('Visualization of the dataset'))


% Specified in plot order
legend('y = 1', 'y = 0')
hold off;

% Add Polynomial Features

X = mapFeature(X(:,1), X(:,2));
X_test = mapFeature(X_test(:,1), X_test(:,2));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);



%====without regularization==========

%set lambda=0
lambda=0;

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionRegL2(t, X, y, lambda)), initial_theta, options);

% Plot Boundary
plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('Without regularization'))

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

% Compute accuracy on our training set
p = predict(theta, X);
p_test=predict(theta,X_test);
fprintf('without regularization (lambda=0) \n');
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('test Accuracy: %f\n\n\n', mean(double(p_test == y_test)) * 100);



%====L1 regularization==========

%set lambda
lambda=0.01;


% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionRegL1(t, X, y, lambda)), initial_theta, options);

% Plot Boundary
plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('L1 regularization \nlambda = %g', lambda))

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

% Compute accuracy on our training set
p = predict(theta, X);
p_test=predict(theta,X_test);

fprintf('L1 regularization \n');
fprintf('lambda : %f \n',lambda);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('test Accuracy: %f\n\n\n', mean(double(p_test == y_test)) * 100);


%====L2 regularization==========

%set lambda
lambda=1;

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionRegL2(t, X, y, lambda)), initial_theta, options);

% Plot Boundary
plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('L2 regularization \nlambda = %g', lambda))

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

% Compute accuracy on our training set
p = predict(theta, X);
p_test=predict(theta,X_test);

fprintf('L2 regularization \n');
fprintf('lambda : %f \n',lambda);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('test Accuracy: %f\n\n\n', mean(double(p_test == y_test)) * 100);


















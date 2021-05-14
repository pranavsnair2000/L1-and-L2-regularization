function [J, grad] = costFunctionRegL2(theta, X, y, lambda)

m = length(y);

J = 0;
grad = zeros(size(theta));


z=X*theta;
h=sigmoid(z);

%cost
J=((1/m)*sum(-y.*log(h) -(1-y).*log(1-h))) + ((lambda/(2*m))*sum(theta(2:size(theta),1).^2));

%gradient
grad(1,1)=((1/m)*sum((h.-y).*X)'(1,1));
grad(2:size(theta),1)=((1/m)*sum((h.-y).*X)'(2:size(theta),1))+((lambda/m)*theta(2:size(theta),1));


end

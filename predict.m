function p = predict(theta, X)

m = size(X, 1); % Number of training examples

p = zeros(m, 1);



z=X*theta;
h=sigmoid(z);
for i=1:m
  if h(i)>=0.5
    p(i)=1;
  endif
endfor






end

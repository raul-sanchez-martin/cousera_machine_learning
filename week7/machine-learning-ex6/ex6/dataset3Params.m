function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%



poss_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
errors = zeros(length(poss_values), length(poss_values));

for C_idx = 1:length(poss_values)
  for sigma_idx = 1:length(poss_values)
    curr_C = poss_values(C_idx);
    curr_sigma = poss_values(sigma_idx);
    model = svmTrain(X, y, curr_C, @(x1, x2) gaussianKernel(x1, x2, curr_sigma));
    predictions = svmPredict(model, Xval);
    curr_error = mean(double(predictions ~= yval));
    errors(C_idx, sigma_idx) = curr_error; 
  endfor
endfor

[minval, C_idx_min] = min(min(errors,[],2));
[minval, sigma_idx_min] = min(min(errors,[],1));

C = poss_values(C_idx_min)
sigma = poss_values(sigma_idx_min)

% =========================================================================

end

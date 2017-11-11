function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K --> number of centroids
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for obs_idx = 1:size(X,1)
  curr_dist = zeros(1, K);
  for centr_idx = 1:K
    curr_obs = X(obs_idx,:);
    curr_centroid = centroids(centr_idx, :);
    dist = sqrt(sum((curr_obs .- curr_centroid) .^ 2));
    curr_dist(1,centr_idx) = dist;
  endfor
  idx(obs_idx) = find(curr_dist == min(curr_dist));
endfor


% =============================================================

end


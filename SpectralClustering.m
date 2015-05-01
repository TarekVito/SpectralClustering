%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Spectral Clustering based on Random Walk (Normalized laplacian Matrix)
%Input : 
%affinity : n*n matrix define the similarities between each instance in the
%           data
%K (optional) : K Expected clusters. (If it's not provided, clustering will
%               be based on Eigenvalues
%
%Output : 
%KEigenVectors : The K eigenvectors that can be used to cluster all the
%data
%For example you can use it this way:
%   KEigenVectors = SpectralClustering(affinity,K)
%   [IDX,C] = kmeans(KEigenVectors,K)
%   %Where IDX contains the cluster indices of each instance. 

function [KEigenVectors]= SpectralClustering(affinity,K)

sz = size(affinity,1);
D = zeros(sz,sz);
for i=1:sz
    D(i,i) = sum(affinity(i,:));
end

L = D - affinity;
[eigVec, eigVal] = eig( inv(D) * L);

cEigVal =  circshift(diag(eigVal),1);
[~, idx] = max(cEigVal - diag(eigVal));
if ~exist('K','var')
    KEigenVectors = eigVec(:,idx:size(eigVec,1)-1);
else
    KEigenVectors = eigVec(:,size(eigVec,1)-K:size(eigVec,1)-1);
end
end

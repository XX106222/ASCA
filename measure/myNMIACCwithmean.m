function [resmax,best_indx]= myNMIACCwithmean(U,Y,numclass)

stream = RandStream.getGlobalStream;
reset(stream);
U_normalized = U ./ repmat(sqrt(sum(U.^2, 2)), 1,size(U,2));
maxIter = 50;
all_indx = zeros(length(Y), maxIter);

for iter = 1:maxIter
    indx = litekmeans(U_normalized,numclass,'MaxIter',100, 'Replicates',1);
%     indx = kmeans(U_normalized,numclass,'MaxIter',100, 'Replicates',1);
    indx = indx(:);
    all_indx(:, iter) = indx;
    result(iter,:) = Clustering8Measure(Y,indx);
end
resmax = max(result,[],1);
[~, best_iter] = max(mean(result, 2));
best_indx = all_indx(:, best_iter);
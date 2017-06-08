% variables
k = 2 % number of clusters for k-means

% create dummy data
x = [0.2,0.3,0.4,0.5,4.4,4.5,4.5,4.9,4.9,5.1];
y = [3.2,3.3,3.2,3.4,2.3,2.4,2.5,2.3,2.4,2.5];
labels = [1,1,1,1,1,1,1,-1,-1,-1];
data = [x',y'];
scatter(data(:,1), data(:,2), 30,labels);

% layer 1 - run svm on whole dataset
lyr1_model = fitcsvm(data, labels);
% end of layer 1

% layer 2 - cluster data and run SVM on each cluster.
[kmeans_label, centroids] = kmeans(data, k);
scatter(data(:,1), data(:,2), 30,kmeans_label);
% loop over each cluster and perform SVM if cluster contains more than one
% label

%initialize arrays to hold centroid and model information
model = cell(size(k));
cluster_data = cell(size(k));
for i=1:k
    % get index of elements in cluster
   idx = find(kmeans_label == i);
   cluster_data{i} = data(idx, :);
   %if cluster contains one label only, associate label with cluster
   if length(unique(labels(idx))) == 1
       lyr2_label = labels(idx(1));
       model{i} = labels(idx(1));
   % else train SVM on cluster
   else
       model{i} = fitcsvm(data(idx, :), labels(idx));
   end
end

l = layer;
l.centroid_pos = centroids;
l.centroid_class = model;
l.centroid_data = cluster_data;
% end of layer 2

% TODO - plot SVM to verify correct boundary
% TODO - test with different values of K
% TODO - test on real dataset
% TODO - create a function to automate layering
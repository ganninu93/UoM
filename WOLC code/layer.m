classdef layer
    % This object stores the data related to different clusters within a
    % layer. This data includes the centroid locations and the SVM model or
    % label depending on whether or not the cluster contains multiple or a
    % single class respectively
    
    properties
        centroid_pos % A matrix of centroid coordinates
        centroid_class % An array containing the classification (label or SVM) of the centroids
        centroid_data % The coordinates of the datapoints within the specific clusters
    end
    
    methods
    end
    
end


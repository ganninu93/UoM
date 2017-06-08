% Maps a set of string labels to a numeric value 
%Input:
%   stringLabels - An Nx1 cell matrix containing the labels in string
%   format
%Output:
%   numericLabels - An Nx1 matrix containing the labels in integer format

function [ numericLabels ] = convertStringToIntLabels( stringLabels )
    uniqueLabels = unique(stringLabels);
    numericLabels = zeros(size(stringLabels));
    for i = 1:numel(uniqueLabels)
       numericLabels(strcmp(stringLabels, uniqueLabels(i))) = i;
    end
end


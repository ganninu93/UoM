function [trainingData, trainingLabels, testingData, testingLabels] = generateFold(data, labels, trainRatio)
    uniqueLabels = numel(unique(labels));
    classIdx = cell(uniqueLabels,1);
    % split data points into cells according to their class
    for label = 1:uniqueLabels
       classIdx{label} = find(labels == label); 
    end
    
    trainingIdx = [];
    testingIdx = [];
    % select n random datapoints from each class to be the training sample.
    % The number of datapoints selected (n) depends on the trainRatio
   for label = 1 : uniqueLabels
       sampleSize = numel(classIdx{label});
       trainIdx = randperm(sampleSize, ceil(sampleSize*trainRatio));
       trainingIdx = [trainingIdx;classIdx{label}(trainIdx)];
       testingIdx = [testingIdx;classIdx{label}(~ismember(1:sampleSize,trainIdx))];
   end
   
   trainingData = data(trainingIdx,:);
   testingData = data(testingIdx,:);
   trainingLabels = labels(trainingIdx);
   testingLabels = labels(testingIdx);
end


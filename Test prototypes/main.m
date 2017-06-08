
% pre-process iris dataset
labels = convertStringToIntLabels(iris_labels);
data = iris_data;

% split data set into 80% training and 20% testing data
randomizedIdx = randperm(size(data,1));
trainSize = ceil(size(data,1)*0.8);
trainData = data(randomizedIdx(1:trainSize),:);
trainLabels = labels(randomizedIdx(1:trainSize),:);
testData = data(randomizedIdx(trainSize+1:end),:);
testLabels = labels(randomizedIdx(trainSize+1:end),:);

%codeMatrix = generateOneVsAllMatrix(numel(unique(labels)));

% generate test code matrix
codeMatrix = [1, 1, 0, 1;... 
              0, 1, 1,-1;...
             -1,-1,-1, 0];

% preallocate cell to store SVM models
models = cell(size(codeMatrix,2),1);

%%%%%%%%%%%%%%%%%%
% Learning phase %
%%%%%%%%%%%%%%%%%%

% Extracts the data to be compared and reassigns it new classes according
% those specified by the code matrix
for col = 1:size(codeMatrix,2)
    % identify the positive and negative classes. In case of ternary codes,
    % classes labels with 0 in the code matrix will be ignored
    posClass = find(codeMatrix(:,col) == 1);
    negClass = find(codeMatrix(:,col) == -1);
    posData = trainData(find(ismember(trainLabels,posClass)),:);
    negData = trainData(find(ismember(trainLabels,negClass)),:);
    tmpData = [posData; negData];
    tmpLabels = [zeros(size(posData,1),1); ones(size(negData,1),1)];
    models{col} = fitcsvm(tmpData,tmpLabels);
end

%%%%%%%%%%%%%%%%%%%%
% Validation phase %
%%%%%%%%%%%%%%%%%%%%

% Use the models built during the training phase to predict data
predictions = zeros(size(testData,1), size(codeMatrix,2));
for col = 1:size(codeMatrix,2)
   predictions(:,col) = predict(models{col}, testData);
end

% the output of the function predict is in terms of ones and zeros.
% the zeros need to be converted to -1 and negated to match our convention
predictions(predictions==0) = -1;
predictions = predictions .* -1;

% build H matrix which holds the performance of the dichotomizers
performanceMat = zeros(size(codeMatrix));
for col = 1:size(codeMatrix,2)
   for row = 1:size(codeMatrix,1)
      if codeMatrix(row,col) ~= 0
          expectedValue = codeMatrix(row, col);
          classIdx = find(testLabels == row);
          predictedValue = predictions(classIdx, col);
          performanceMat(row, col) = sum(predictedValue == expectedValue)/numel(classIdx);
      end
   end
end

% build weight matrix by normalising H matrix so that each row sums up to 1
weightMat = zeros(size(codeMatrix));
for row = 1:size(codeMatrix,1)
    weightMat(row,:) = (1/sum(performanceMat(row,:))) .* performanceMat(row, :);
end

% compare predicted codeword to code matrix and find the closest using
% hamming distance
finalPrediction = zeros(size(data,1),1);
for prediction = 1:size(predictions,1)
   hammingDist =  pdist2(codeMatrix,predictions(prediction, :), 'hamming');
   [mn, closestClass] = min(hammingDist);
   finalPrediction(prediction) = closestClass;
end

accuracy = sum((finalPrediction==labels))/numel(labels)

% Use MATLAB's inbuilt function
ecoc = fitcecoc(data, labels, 'Coding', 'onevsall');
matlabEcocPred = predict(ecoc,data, 'BinaryLoss', 'hamming');
accuracyMatlab = sum((matlabEcocPred==labels))/numel(labels)
% train and predict using the fano metric and loss weighted technique for
% 10 times, each time calculating the accuracy. Average out the accuracies
% and see which method is superior

numFolds = 10;

% pre-process iris dataset
%labels = convertStringToIntLabels(ecoli_labels);
labels = glass_labels;
data = glass_data;

% split the labels into folds of approximately equal size and containing an
% equal distribution of classes.
partitions = cvpartition(numel(labels),'kfold',numFolds);

% generateCodeMatrix
codeMatrix = generateOneVsOneMatrix(numel(unique(labels)));

% create an accuracies matrix where the rows correspond to the different
% methods and the columns to the different test
accuracies = zeros(3, numFolds);

for foldNo = 1:numFolds
    trainIdx = training(partitions, foldNo);
    trainData = data(trainIdx,:);
    trainLabels = labels(trainIdx,:);
    testData = data(~trainIdx,:);
    testLabels = labels(~trainIdx,:);
    
    numTrainLabels = numel(unique(trainLabels))
    numTestLabels = numel(unique(testLabels))
    
    %%%%%%%%%%%%%%%%%%
    % Encoding phase %
    %%%%%%%%%%%%%%%%%%

    % use Matlab's inbuilt ECOC function for learning
    ecocMdl = fitcecoc(trainData, trainLabels, 'Coding', codeMatrix);
    
    %%%%%%%%%%%%%%%%%%
    % Decoding phase %
    %%%%%%%%%%%%%%%%%%
    accuracies(1,foldNo) = trainAndPredictFano(trainData, trainLabels, testData, testLabels, codeMatrix, ecocMdl);
    accuracies(2,foldNo) = trainAndPredictFanoV2(trainData, trainLabels, testData, testLabels, codeMatrix, ecocMdl);
    accuracies(3,foldNo) = trainAndPredictLossWeighted(trainData, trainLabels, testData, testLabels, codeMatrix, ecocMdl);
end

avgAcc = sum(accuracies, 2)/numFolds;
disp(strcat('Fano metric = ', num2str(avgAcc(1))));
disp(strcat('Fano metric V2 = ', num2str(avgAcc(2))));
disp(strcat('Loss weighted = ', num2str(avgAcc(3))));

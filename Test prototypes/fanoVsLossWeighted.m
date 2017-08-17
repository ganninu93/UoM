% train and predict using the fano metric and loss weighted technique for
% 10 times, each time calculating the accuracy. Average out the accuracies
% and see which method is superior
tic;
numFolds = 10;

% pre-process iris dataset
%labels = convertStringToIntLabels(ecoli_labels);
labels = glass_labels;
data = glass_data;

% Setup training params
Parameters.coding='ECOCONE';
Parameters.decoding='ELW';
Parameters.base='SVM';
Parameters.base_params.iterations=50;
% Setup testing params
Parameters.base_test='SVMtest';

% create an accuracies matrix where the rows correspond to the different
% methods and the columns to the different test
accuracies = zeros(4, numFolds);

for foldNo = 1:numFolds
    % print fold number to keep track of progress
    disp(strcat('Starting fold...', num2str(foldNo)));
    
    [trainData, trainLabels, testData, testLabels] = generateFold(data, labels, 0.9);
    
    %%%%%%%%%%%%%%%%%%
    % Encoding phase %
    %%%%%%%%%%%%%%%%%%

    [Classifiers,Parameters]=ECOCTrain(trainData,trainLabels,Parameters);

    % use Matlab's inbuilt ECOC function for learning
    codeMatrix = Parameters.ECOC;
    ecocMdl = fitcecoc(trainData, trainLabels, 'Coding', codeMatrix);
    
    % Use the models built during the training phase to predict data
    cellPredictions = cellfun(@predict, ecocMdl.BinaryLearners, repmat({trainData},size(codeMatrix,2),1),'UniformOutput', false);
    predictions = [cellPredictions{:}]; % convert cells back to matrix
    %predictions = cell2mat(cellfun(@predict, ecocMdl.BinaryLearners', repmat({trainData},1,numel(ecocMdl.BinaryLearners)), 'UniformOutput', false));

    % the output of the function predict is in terms of ones and zeros.
    % the zeros need to be converted to -1 and negated to match our convention
    predictions(predictions==0) = -1;
    
    % build H matrix which holds the performance of the dichotomizers
    performanceMat = zeros(size(codeMatrix));
    for col = 1:size(codeMatrix,2)
        for row = 1:size(codeMatrix,1)
            if codeMatrix(row,col) ~= 0
                expectedValue = codeMatrix(row, col);
                classIdx = find(trainLabels == row);
                predictedValue = predictions(classIdx, col);
                performanceMat(row, col) = sum(predictedValue == expectedValue)/numel(classIdx);
            end
        end
    end
    
    % calculate probability that 1 occurs in places where codeMatrix=0
    % the remaining locations are marked with a -1 for easier identification
    tailProbOfOne = ones(size(codeMatrix))*-1;
    for row = 1:size(codeMatrix,1)
        for col = 1:size(codeMatrix,2)
            if (codeMatrix(row, col) == 0)
                classIdx = find(trainLabels == row);
                predictedValue = predictions(classIdx, col);
                tailProbOfOne(row, col) = sum(predictedValue == 1)/numel(predictedValue);
            end
        end
    end
    
    % predict test data
    testPred = zeros(size(testData,1),size(codeMatrix, 2));
    for col = 1:size(codeMatrix, 2)
        testPred(:,col) = predict(ecocMdl.BinaryLearners{col}, testData);
    end 
    
    %%%%%%%%%%%%%%%%%%
    % Decoding phase %
    %%%%%%%%%%%%%%%%%%
    accuracies(1,foldNo) = trainAndPredictFano(performanceMat, tailProbOfOne, predictions, testPred, trainLabels, testData, testLabels, codeMatrix, ecocMdl);
    accuracies(2,foldNo) = trainAndPredictFanoV2(performanceMat, tailProbOfOne, predictions, testPred, trainLabels, testData, testLabels, codeMatrix, ecocMdl);
    accuracies(3,foldNo) = trainAndPredictFanoV3(performanceMat, tailProbOfOne, predictions, testPred, trainLabels, testData, testLabels, codeMatrix, ecocMdl);
    accuracies(4,foldNo) = trainAndPredictLossWeighted(performanceMat, testData, testLabels, testPred, codeMatrix, ecocMdl);
    accuracies(:, foldNo)'
end
avgAcc = mean(accuracies,2);
disp(strcat('Fano metric V1= ', num2str(avgAcc(1))));
disp(strcat('Fano metric V2 = ', num2str(avgAcc(2))));
disp(strcat('Fano metric V3 = ', num2str(avgAcc(3))));
disp(strcat('Loss weighted = ', num2str(avgAcc(4))));
toc;

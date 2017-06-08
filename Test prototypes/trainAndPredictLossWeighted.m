function accuracy = trainAndPredictLossWeighted(trainData, trainLabels, testData, testLabels, codeMatrix, ecocMdl)

    %%%%%%%%%%%%%%%%%%%%
    % Validation phase %
    %%%%%%%%%%%%%%%%%%%%

    % The following section uses the models generated during the training phase
    % to predict the training data. The accuracy for each model-class pair will
    % be calculated and a weight matrix will be generated based on these
    % accuracies

    % Use the models built during the training phase to predict data
    predictions = zeros(size(trainData,1), size(codeMatrix,2));
    for col = 1:size(codeMatrix,2)
        predictions(:,col) = predict(ecocMdl.BinaryLearners{col}, trainData);
    end

    % the output of the function predict is in terms of ones and zeros.
    % the zeros need to be converted to -1 and negated to match our convention
    predictions(predictions==0) = -1;
    %predictions = predictions .* -1;

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

    % build weight matrix by normalising H matrix so that each row sums up to 1
    weightMat = zeros(size(codeMatrix));
    for row = 1:size(codeMatrix,1)
        weightMat(row,:) = (1/sum(performanceMat(row,:))) .* performanceMat(row, :);
    end

    %%%%%%%%%%%%%%%%%
    % Testing Phase %
    %%%%%%%%%%%%%%%%%

    % the following section is responsible for predicting the class of unseen
    % data. This is carried out by iterating over every data point in the test
    % dataset, comparing it against the code matrix to calculate the
    % exponential loss and cross multiplying the results with the weight
    % matrix. The row whose sum of weightred loss is the smallest will be
    % declared the winning class.

    predictions = zeros(numel(testLabels),1);
    for dataPt = 1:numel(testLabels)
        % get predictions from every dichotomy
        binaryPred = zeros(1,size(codeMatrix, 2));
        for col = 1:size(codeMatrix, 2)
            binaryPred(col) = predict(ecocMdl.BinaryLearners{col}, testData(dataPt,:));
        end

        % Calculate weighted loss
        weightedLoss = zeros(size(codeMatrix, 1),1);
        for row = 1:size(codeMatrix, 1)
            expLoss = exp(-binaryPred .* codeMatrix(row, :));
            weightedLoss(row) = sum(weightMat(row,:) .* expLoss);
        end

        % select class with lowest loss
        [~, predictions(dataPt)] = min(weightedLoss);
    end

    % calculate accuracy
    accuracy = sum((predictions==testLabels))/numel(testLabels);
end


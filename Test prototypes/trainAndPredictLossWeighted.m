function accuracy = trainAndPredictLossWeighted(performanceMat, testData, testLabels,  testPred, codeMatrix, ecocMdl)

    %%%%%%%%%%%%%%%%%%%%
    % Validation phase %
    %%%%%%%%%%%%%%%%%%%%

    % The following section uses the models generated during the training phase
    % to predict the training data. The accuracy for each model-class pair will
    % be calculated and a weight matrix will be generated based on these
    % accuracies

    % build weight matrix by normalising H matrix so that each row sums up to 1
    weightMat = repmat(1./sum(performanceMat,2),1,size(performanceMat,2)) .* performanceMat;

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
        % Calculate weighted loss
        weightedLoss = zeros(size(codeMatrix, 1),1);
        for row = 1:size(codeMatrix, 1)
            expLoss = exp(-testPred(dataPt, :) .* codeMatrix(row, :));
            weightedLoss(row) = sum(weightMat(row,:) .* expLoss);
        end

        % select class with lowest loss
        [~, predictions(dataPt)] = min(weightedLoss);
    end

    % calculate accuracy
    accuracy = sum((predictions==testLabels))/numel(testLabels);
end

function accuracy = trainAndPredictFano(trainData, trainLabels, testData, testLabels, codeMatrix, ecocMdl)
    numLabels = numel(unique(trainLabels));

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
    
    %predictions = cell2mat(cellfun(@predict, ecocMdl.BinaryLearners', repmat({trainData},1,numel(ecocMdl.BinaryLearners)), 'UniformOutput', false));

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

    %%%%%%%%%%%%%%%%%
    % Testing Phase %
    %%%%%%%%%%%%%%%%%

    testPredictions = zeros(numel(testLabels), 1);

    % calculate apriori probability of class distribution based on training
    % labels
    aprioriProb = zeros(numLabels,1);
    for label = 1:numLabels
        aprioriProb(label) = sum(trainLabels==label)/numel(trainLabels);
    end

    % predict test data points
    for dataPt = 1:size(testData,1)
        % get predictions from every dichotomy
        binaryPred = zeros(1,size(codeMatrix, 2));
        for col = 1:size(codeMatrix, 2)
            binaryPred(col) = predict(ecocMdl.BinaryLearners{col}, testData(dataPt,:));
        end

        fanoProb = ones(size(codeMatrix));
        for row = 1:size(codeMatrix,1)
            for col = 1:size(codeMatrix,2)
                % calculate cross over probability
                if(codeMatrix(row,col) ~= 0)
                    if(binaryPred(col) == codeMatrix(row, col))
                        fanoProb(row, col) = performanceMat(row, col);
                    else
                        fanoProb(row, col) = 1-performanceMat(row, col);
                    end
                % else calculate tail probability
                else
                    if(binaryPred(col) == 1)
                        fanoProb(row, col) = tailProbOfOne(row, col);
                    else
                        fanoProb(row, col) = 1 - tailProbOfOne(row, col);
                    end
                end
            end
        end

        [~, testPredictions(dataPt)] = max(aprioriProb .* prod(fanoProb,2));
    end

    accuracy = sum(testPredictions == testLabels)/numel(testLabels);
end


function accuracy = trainAndPredictFano(performanceMat, tailProbOfOne, predictions, testPred, trainLabels, testData, testLabels, codeMatrix, ecocMdl)
    numLabels = numel(unique(trainLabels));

    testPredictions = zeros(numel(testLabels), 1);

    % calculate apriori probability of class distribution based on training
    % labels
    aprioriProb = zeros(numLabels,1);
    for label = 1:numLabels
        aprioriProb(label) = sum(trainLabels==label)/numel(trainLabels);
    end

    % predict test data points
    for dataPt = 1:size(testData,1)
        fanoProb = ones(size(codeMatrix));
        for row = 1:size(codeMatrix,1)
            for col = 1:size(codeMatrix,2)
                % calculate cross over probability
                if(codeMatrix(row,col) ~= 0)
                    if(testPred(dataPt, col) == codeMatrix(row, col))
                        fanoProb(row, col) = performanceMat(row, col);
                    else
                        fanoProb(row, col) = 1-performanceMat(row, col);
                    end
                % else calculate tail probability
                else
                    if(testPred(dataPt, col) == 1)
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


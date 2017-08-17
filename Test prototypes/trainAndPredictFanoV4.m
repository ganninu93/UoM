function accuracy = trainAndPredictFanoV4(performanceMat, tailProbOfOne, predictions, testPred, trainLabels, testData, testLabels, codeMatrix, ecocMdl)
    numLabels = numel(unique(trainLabels));

    % calculate apriori probability of class distribution based on training
    % labels
    aprioriProb = zeros(numLabels,1);
    for label = 1:numLabels
        aprioriProb(label) = sum(trainLabels==label)/numel(trainLabels);
    end
    
    % calculate probability that dichotomy outputs a 1 based on entire
    % training dataset
    probOfOne = sum(predictions==1)/size(predictions,1);
    
    % Calculate cross-over probability P(y|t)
    confusionMatrices = cell(size(codeMatrix,2),1);
    for col = 1:size(codeMatrix,2)
        % determine classes used by dichotomy
        posClass = find(codeMatrix(:,col) == 1);
        negClass = find(codeMatrix(:,col) == -1);
        % find samples used to train dichotomy
        posIdx = find(trainLabels == posClass);
        negIdx = find(trainLabels == negClass);
        % retreive result outputted by dichotomy when the samples where
        % inputted
        posPred = predictions(posIdx,col);
        negPred = predictions(negIdx, col);
        % build confusion matrix
        input = [ones(numel(posIdx),1); -1*ones(numel(negIdx),1)];
        output = [posPred;negPred];
        confusionMatrices{col} = confusionmat(input,output)/numel(input);
    end
    
    % Classify new data
    for dataPt = 1:numel(testLabels)
        crossOverProbTail = zeros(size(codeMatrix,1),1);
        crossOverProbNonTail = ones(size(codeMatrix,1),1);
        for row = 1:size(codeMatrix,1)
            zeroIdx = find(codeMatrix(row,:)==0);
            nonZeroIdx = find(codeMatrix(row,:)~=0);
            
            % calculate cross-over for non-tail bits
            confMatOrder = [-1,1];
            input = codeMatrix(row, nonZeroIdx);
            output = predictions(dataPt, nonZeroIdx);
            for col = 1:numel(input)
               confMat = confusionMatrices{nonZeroIdx(col)};
               crossOverProbNonTail(row) = crossOverProbNonTail(row) * confMat(find(input(col)==confMatOrder), find(output(col)==confMatOrder));
            end
            
            % calculate probability of tail bits
            Qt1 = tailProbOfOne(row, zeroIdx);
            QtNot1 = 1-Qt1;
            crossOverFromOne = ones(1,numel(zeroIdx));
            crossOverFromMinusOne = ones(1,numel(zeroIdx));
            tailOutput = predictions(dataPt, zeroIdx);
            for col = 1:numel(tailOutput)
               confMat = confusionMatrices{zeroIdx(col)};
               crossOverFromMinusOne(col) = confMat(1,find(tailOutput(col)==confMatOrder));
               crossOverFromOne(col) = confMat(2, find(tailOutput(col)==confMatOrder));
            end
            
            probOne = Qt1 .* crossOverFromOne;
            probMinusOne = QtNot1 .* crossOverFromMinusOne;
            crossOverProbTail(row) = prod(probOne + probMinusOne);
        end
        fanoMetric = aprioriProb .* crossOverProbNonTail .* crossOverProbTail;
        [~, testPredictions(dataPt)] = max(fanoMetric);
    end
    accuracy = sum(testPredictions' == testLabels)/numel(testLabels);
end
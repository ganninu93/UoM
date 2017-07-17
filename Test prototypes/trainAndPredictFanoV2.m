% This implementation takes the probability Qt to be the probability that a
% dichotomy produces a specific value irrispective of the class being
% passed. The probability Pr(y|t) is the cross over tail probability
function accuracy = trainAndPredictFanoV2(performanceMat, tailProbOfOne, predictions, testPred, trainLabels, testData, testLabels, codeMatrix, ecocMdl)
    numLabels = numel(unique(trainLabels));

    % calculate probability that dichotomy outputs a 1 based on entire
    % training dataset
    probOfOne = sum(predictions==1)/size(predictions,1);
    
    % calculate apriori probability of class distribution based on training
    % labels
    aprioriProb = zeros(numLabels,1);
    for label = 1:numLabels
        aprioriProb(label) = sum(trainLabels==label)/numel(trainLabels);
    end
    
    %%%%%%%%%%%%%%%%%
    % Testing Phase %
    %%%%%%%%%%%%%%%%%
    
    testPredictions = zeros(numel(testLabels), 1);
    
    % perform tail calculations
    Qt = cell(size(codeMatrix,1),1);
    for row = 1:size(codeMatrix,1)
        zeroIdx = find(codeMatrix(row,:)==0);
        numZerosInRow = numel(zeroIdx);
        %generate all possible combinations for zero locations
        binComb = dec2bin([0:(2^numZerosInRow)-1]);
        % Calculate probability that given tail is generated based on
        % dichotomy's performance on entire training data set - Qt
        bitProb = zeros(size(binComb,1),size(binComb,2));
        for bit = 1:size(binComb,2)
            bitProb(:,bit) = ((1-str2num(binComb(:,bit)))*(1-probOfOne(zeroIdx(bit))))+str2num(binComb(:,bit))*probOfOne(zeroIdx(bit));
        end
        Qt{row} = prod(bitProb,2);
    end

    % predict test data points
    for dataPt = 1:size(testData,1)
        crossOverProbTail = zeros(size(codeMatrix,1),1);
        crossOverProb = ones(size(codeMatrix,1),1);
        for row = 1:size(codeMatrix,1)
            zeroIdx = find(codeMatrix(row,:)==0);
            nonZeroIdx = find(codeMatrix(row,:)~=0);
            %Calculate cross over prob for non tail elements
            for idx = 1:numel(nonZeroIdx)
                if(testPred(dataPt, nonZeroIdx(idx)) == codeMatrix(row, nonZeroIdx(idx)))
                    crossOverProb(row) = crossOverProb(row)*performanceMat(row, nonZeroIdx(idx));
                else
                    crossOverProb(row) = crossOverProb(row)*(1-performanceMat(row, nonZeroIdx(idx)));
                end
            end
            %Calculate cross over prob for all tails
            numZerosInRow = numel(zeroIdx);
            %generate all possible combinations for zero locations
            binComb = dec2bin([0:(2^numZerosInRow)-1]);
            % calculate cross over prob
            crossOverZerosRow = repmat(tailProbOfOne(row, zeroIdx), size(binComb,1), 1);
            crossOverOne = (binComb == '1') .* crossOverZerosRow;
            crossOverZero = (binComb == '0') .* (1-crossOverZerosRow);
            % add probabilities together to get a combination of both ones
            % and zeros
            crossOverCombined = crossOverOne + crossOverZero;
            % multiply the probabilities for each binary combination
            % multiply the product by Qt
            % perform summation
            crossOverProbTail(row) = sum(sum(crossOverCombined,2) .* Qt{row});
            
        end
        
        fanoMetric = aprioriProb .* crossOverProb .* crossOverProbTail;
        [~, testPredictions(dataPt)] = max(fanoMetric);
    end

    accuracy = sum(testPredictions == testLabels)/numel(testLabels);
end


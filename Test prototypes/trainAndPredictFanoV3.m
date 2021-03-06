% This implementation calculates the probability that a tail is generated
% (Qt), based on the predictions produces when specific classes are
% inputted. E.g when calculating the tails of the first row, only the
% probabilities of the first class are considered.
function accuracy = trainAndPredictFanoV3(performanceMat, tailProbOfOne, predictions,  testPred, trainLabels, testData, testLabels, codeMatrix, ecocMdl)
    numLabels = numel(unique(trainLabels));
    
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
    
    Qt = cell(size(codeMatrix,1),1);
    for row = 1:size(codeMatrix,1)
        % locate position of zeros
        zeroIdx = find(codeMatrix(row,:)==0);
        numZerosInRow = numel(zeroIdx);
        if numZerosInRow >0
            % calculate probability that zero positions output a value of 1 for
            % the current class (row)
            classPredictions = predictions(find(trainLabels==row),zeroIdx);
            probOfOne = sum(classPredictions == 1)/size(classPredictions,1);

            %generate all possible combinations for zero locations
            binComb = dec2bin([0:(2^numZerosInRow)-1]);
            % Calculate probability that given tail is generated based on
            % dichotomy's the performance of a specific class - Qt
            temp = zeros(size(binComb,1),size(binComb,2));
            for bit = 1:size(binComb,2)
                temp(:,bit) = ((1-str2num(binComb(:,bit)))*(1-probOfOne(bit)))+str2num(binComb(:,bit))*probOfOne(bit);
            end
            Qt{row} = prod(temp,2);
        end
    end

    % predict test data points
    for dataPt = 1:size(testData,1)               
        crossOverProbTail = ones(size(codeMatrix,1),1);
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
            if numZerosInRow > 0
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
        end
        
        fanoMetric = aprioriProb .* crossOverProb .* crossOverProbTail;
        [~, testPredictions(dataPt)] = max(fanoMetric);
    end

    accuracy = sum(testPredictions == testLabels)/numel(testLabels);
end
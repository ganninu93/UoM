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
    
    for row = 1:size(codeMatrix,1)
        % locate position of zeros
        zeroIdx = find(codeMatrix(row,:)==0);
        % calculate probability that zero positions output a value of 1 for
        % the current class (row)
        classPredictions = predictions(find(trainLabels==row),zeroIdx);
        probOfOne = sum(classPredictions == 1)/size(classPredictions,1);
    end

    % predict test data points
    for dataPt = 1:size(testData,1)               
        crossOverProbTail = zeros(size(codeMatrix,1),1);
        crossOverProb = ones(size(codeMatrix,1),1);
        for row = 1:size(codeMatrix,1)
            zeroIdx = find(codeMatrix(row,:)==0);
            nonZeroIdx = find(codeMatrix(row,:)~=0);
            numZerosInRow = numel(zeroIdx);

            %Calculate cross over prob for non tail elements
            for idx = 1:numel(nonZeroIdx)
                if(testPred(dataPt, nonZeroIdx(idx)) == codeMatrix(row, nonZeroIdx(idx)))
                    crossOverProb(row) = crossOverProb(row)*performanceMat(row, nonZeroIdx(idx));
                else
                    crossOverProb(row) = crossOverProb(row)*(1-performanceMat(row, nonZeroIdx(idx)));
                end
            end
            
            %generate all possible combinations for zero locations
            %if bin num contains more than 10 bits, the process is performed in
            %batches
            maxBinNum = (2^numZerosInRow)-1;
            batchStart = [0:1000000:maxBinNum];
            if maxBinNum < 999999
                batchEnd = maxBinNum;
            else
                batchEnd = [999999:1000000:maxBinNum,maxBinNum];
            end
            
            for batchIdx = 1:numel(batchStart)
                binComb = dec2bin([batchStart(batchIdx):batchEnd(batchIdx)]);
                % Calculate probability that given tail is generated based on
                % dichotomy's the performance of a specific class - Qt
                temp = zeros(size(binComb,1),size(binComb,2));
                for bit = 1:size(binComb,2)
                    temp(:,bit) = ((1-str2num(binComb(:,bit)))*(1-probOfOne(bit)))+str2num(binComb(:,bit))*probOfOne(bit);
                end
                Qt = prod(temp,2);
                
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
                crossOverProbTail(row) = crossOverProbTail(row) + sum(sum(crossOverCombined,2) .* Qt);
            end
        end
        
        fanoMetric = aprioriProb .* crossOverProb .* crossOverProbTail;
        [~, testPredictions(dataPt)] = max(fanoMetric);
    end

    accuracy = sum(testPredictions == testLabels)/numel(testLabels);
end


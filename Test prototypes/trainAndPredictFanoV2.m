% This implementation takes the probability Qt to be the probability that a
% dichotomy produces a specific value irrispective of the class being
% passed. The probability Pr(y|t) is the cross over tail probability
function accuracy = trainAndPredictFanoV2(performanceMat, tailProbOfOne, predictions, testPred, trainLabels, testData, testLabels, codeMatrix, ecocMdl)
    numLabels = numel(unique(trainLabels));

    %%%%%%%%%%%%%%%%%%%%
    % Validation phase %
    %%%%%%%%%%%%%%%%%%%%

    % The following section uses the models generated during the training phase
    % to predict the training data. The accuracy for each model-class pair will
    % be calculated and a weight matrix will be generated based on these
    % accuracies

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
    sumTailProb = zeros(size(codeMatrix,1),1);
    for row = 1:size(codeMatrix,1)
        zeroIdx = find(codeMatrix(row,:)==0);
        numZerosInRow = numel(zeroIdx);
        %generate all possible combinations for zero locations
        binComb = dec2bin([0:(2^numZerosInRow)-1]);
        % Calculate probability that given tail is generated based on
        % dichotomy's performance on entire training data set - Qt
        temp = zeros(size(binComb,1),size(binComb,2));
        for bit = 1:size(binComb,2)
            temp(:,bit) = ((1-str2num(binComb(:,bit)))*(1-probOfOne(zeroIdx(bit))))+str2num(binComb(:,bit))*probOfOne(zeroIdx(bit));
        end
        Qt = prod(temp,2);
        
        % Calculate cross over probability for a given class
        for binNum = 1:size(binComb,1)
            tailProb = 1;
            for bit = 1:size(binComb,2)
                
                if(binComb(binNum, bit) == '1')
                   tailProb = tailProb * tailProbOfOne(row,zeroIdx(bit)) ;
                else
                   tailProb = tailProb * (1-tailProbOfOne(row,zeroIdx(bit)));
                end
            end
            sumTailProb(row) = sumTailProb(row) + tailProb;
        end
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
            for binNum = 1:size(binComb,1)
                for bit = 1:size(binComb,2)
                    if(testPred(dataPt, zeroIdx(bit)) == 1)
                        crossOverProbTail(row) = crossOverProbTail(row)+(Qt(binNum)*tailProbOfOne(row, zeroIdx(bit)));
                    else
                        crossOverProbTail(row) = crossOverProbTail(row)+(Qt(binNum)*(1-tailProbOfOne(row, zeroIdx(bit))));
                    end
                end
            end
        end
        
        fanoMetric = aprioriProb .* crossOverProb .* crossOverProbTail;
        [~, testPredictions(dataPt)] = max(fanoMetric);
    end

    accuracy = sum(testPredictions == testLabels)/numel(testLabels);
end


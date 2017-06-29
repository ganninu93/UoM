% This implementation takes the probability Qt to be the probability that a
% dichotomy produces a specific value irrispective of the class being
% passed. The probability Pr(y|t) is the cross over tail probability
function accuracy = trainAndPredictFanoV2(trainData, trainLabels, testData, testLabels, codeMatrix, ecocMdl)
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
        % get output word i.e get predictions from every dichotomy
        binaryPred = zeros(1,size(codeMatrix, 2));
        for col = 1:size(codeMatrix, 2)
            binaryPred(col) = predict(ecocMdl.BinaryLearners{col}, testData(dataPt,:));
        end
                
        crossOverProbTail = zeros(size(codeMatrix,1),1);
        crossOverProb = ones(size(codeMatrix,1),1);
        for row = 1:size(codeMatrix,1)
            zeroIdx = find(codeMatrix(row,:)==0);
            nonZeroIdx = find(codeMatrix(row,:)~=0);
            %Calculate cross over prob for non tail elements
            for idx = 1:numel(nonZeroIdx)
                if(binaryPred(nonZeroIdx(idx)) == codeMatrix(row, nonZeroIdx(idx)))
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
                    if(binaryPred(zeroIdx(bit)) == 1)
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


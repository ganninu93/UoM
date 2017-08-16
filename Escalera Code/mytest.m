data = optdigits_data;
labels = optdigits_labels;
[trainData, trainLabels, testData, testLabels] = generateFold(data, labels, 0.8);

% Setup training params
Parameters.coding='ECOCONE';
Parameters.decoding='ELW';
Parameters.base='SVM'; % ADA is the name of the function for training using Adaboost
Parameters.base_params.iterations=50; % Required parameter for training using Adaboost. In this case it corresponds to the number of decision stumps
Parameters.show_info = 0;

% Setup testing params
Parameters.base_test='SVMtest';

%training
[Classifiers,Parameters]=ECOCTrain(trainData,trainLabels,Parameters);

%Testing
[Labels,Values,confusion]=ECOCTest(testData,Classifiers,Parameters,testLabels);

accuracy = sum(Labels' == testLabels)/numel(testLabels)
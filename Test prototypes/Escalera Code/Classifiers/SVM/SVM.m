% This function is used to convert the feval function being called in
% Learning.m on line 42 to an svmTrain function (MATLAB inbuilt) call.
function [classifier]=SVM(clase1, clase2, params, test)
    cls1Size = size(clase1,1);
    cls2Size = size(clase2,1);
    
    data = [clase1;clase2];
    labels = [ones(cls1Size,1); (-1*ones(cls2Size,1))];
    classifier = svmtrain(data, labels, 'kernel_function', 'rbf');
end
function [ codeMatrix ] = generateOneVsAllMatrix( numOfClasses )
    codeMatrix = eye(numOfClasses);
    codeMatrix(codeMatrix == 0) = -1;
end


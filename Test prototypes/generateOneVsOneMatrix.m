% this function will generate the one vs one code matrix for the number of 
% classes specified as an input.
function codeMatrix = generateOneVsOneMatrix( numClasses )
    combinations = combnk(1:numClasses, 2);
    codeMatrix = zeros(numClasses, size(combinations,1));
    
    for i = 1:size(combinations,1)
       codeMatrix(combinations(i, 1), i) = 1;
       codeMatrix(combinations(i, 2), i) = -1;
    end
    
    % flip matrix horizontally to confirm with the order used in most
    % papers. This step does not have any impact on the performance
    codeMatrix = fliplr(codeMatrix);
end


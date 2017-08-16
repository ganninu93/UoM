function classes=SVMtest(data,classifier,params)
    classes = svmclassify(classifier, data, 'ShowPlot',false);
end

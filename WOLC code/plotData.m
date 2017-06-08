function plotData(data, labels)
    class0 = find(labels~=1);
    class1 = find(labels==1);
    
    figure;
    hold on;
    ylim([0,5]);
    xlim([0,6]);
    plot(data(class0,1), data(class0,2), 'bx');
    plot(data(class1,1), data(class1,2), 'ro');
    xlabel('X_{1}','FontSize',12,'FontWeight','bold');
    ylabel('X_{2}','FontSize',12,'FontWeight','bold','rot',0);
end


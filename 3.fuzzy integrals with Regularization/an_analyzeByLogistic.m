function [accuarcyList]=an_analyzeByLogistic(U,e,X,Y)
%使用logistic分类分析准确度

    for i=1:size(U,1)
        %当前beta值，为了后面运算，扩增tempBeta,使得和X的维度相同***
        tempBeta=U(i,:);
        tempBeta=repmat(tempBeta,size(X,1),1)
        %当前beta0值***
        tempBeta0=e;
        %使用当前的beta和beta0求出结果***
        tempResult=dot(tempBeta,X,2);
        %求出对应的logistic值，注意这里用./点除***
        tempResult2=1./(1+exp(-tempResult));
        %根据logistic值，将值进行分类，小于0.5的归为0，和大于0.5的归为1***
        [tempResult3]=logisticClassify(tempResult2);
        %把预测的logistic结果tempResult3和真实值Y进行比较****
        [tempResult4,accuarcy]=compare(tempResult3,Y);
        %保存当前accuarcy
        accuarcyList(i,1)=accuarcy;
    end
end

function [tempResult3] =logisticClassify(tempResult2)
    tempResult3=tempResult2;
    tempResult3(find(tempResult3<0.5))=0;
    tempResult3(find(tempResult3>=0.5))=1;
end


function [result,accuarcy] = compare(A,B)
     result=(A==B);
     %样本总数
     amount=size(A,1);
     %判断正确的数量
     accuarcyNum=sum(result);
     %正确率
     accuarcy=accuarcyNum./amount
end
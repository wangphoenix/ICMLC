function [accuarcyList]=an_analyzeByFisher(U,e,Z,trueResult)
%使用fisher分类分析准确度,Z矩阵，trueResult样本真是分类值
%返回值时一个accuarcyList列表，因为U可能多个向量的，多组的（比如用L1/2时候，U是多组的）
    for i=1:size(U,1)
        %当前beta值，为了后面运算，扩增tempBeta,使得和X的维度相同***
        tempBeta=U(i,:);
        tempBeta=repmat(tempBeta,size(Z,1),1)
        %当前beta0值***
        tempBeta0=e(i,:);
        %使用当前的beta和beta0求出结果***
        tempResult=dot(tempBeta,Z,2)+tempBeta0;
        tempResult2=tempResult;
        tempResult2(:,2)=trueResult;
        fitnessResult=an_fitness(tempResult2,2);
        accuarcyList(i,:)=fitnessResult;
    end
end
function [accuarcyList]=an_analyzeByFisher(U,e,Z,trueResult)
%ʹ��fisher�������׼ȷ��,Z����trueResult�������Ƿ���ֵ
%����ֵʱһ��accuarcyList�б���ΪU���ܶ�������ģ�����ģ�������L1/2ʱ��U�Ƕ���ģ�
    for i=1:size(U,1)
        %��ǰbetaֵ��Ϊ�˺������㣬����tempBeta,ʹ�ú�X��ά����ͬ***
        tempBeta=U(i,:);
        tempBeta=repmat(tempBeta,size(Z,1),1)
        %��ǰbeta0ֵ***
        tempBeta0=e(i,:);
        %ʹ�õ�ǰ��beta��beta0������***
        tempResult=dot(tempBeta,Z,2)+tempBeta0;
        tempResult2=tempResult;
        tempResult2(:,2)=trueResult;
        fitnessResult=an_fitness(tempResult2,2);
        accuarcyList(i,:)=fitnessResult;
    end
end
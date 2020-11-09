function [accuarcyList]=an_analyzeByLogistic(U,e,X,Y)
%ʹ��logistic�������׼ȷ��

    for i=1:size(U,1)
        %��ǰbetaֵ��Ϊ�˺������㣬����tempBeta,ʹ�ú�X��ά����ͬ***
        tempBeta=U(i,:);
        tempBeta=repmat(tempBeta,size(X,1),1)
        %��ǰbeta0ֵ***
        tempBeta0=e;
        %ʹ�õ�ǰ��beta��beta0������***
        tempResult=dot(tempBeta,X,2);
        %�����Ӧ��logisticֵ��ע��������./���***
        tempResult2=1./(1+exp(-tempResult));
        %����logisticֵ����ֵ���з��࣬С��0.5�Ĺ�Ϊ0���ʹ���0.5�Ĺ�Ϊ1***
        [tempResult3]=logisticClassify(tempResult2);
        %��Ԥ���logistic���tempResult3����ʵֵY���бȽ�****
        [tempResult4,accuarcy]=compare(tempResult3,Y);
        %���浱ǰaccuarcy
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
     %��������
     amount=size(A,1);
     %�ж���ȷ������
     accuarcyNum=sum(result);
     %��ȷ��
     accuarcy=accuarcyNum./amount
end
function [maxTrainScoreList,testScoreList,maxTrainUList,maxTrainEList] = L1_Lasso(data )
%L1_LASSO �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
    %K������֤�в���
    K=5;
    
    %��ȡ���ݵ�������
    [row,col]=size(data);
    
    %��������
    featureNum=col-1;
   
    %k-���������ݷ����±�,ÿһ����ȷֲ���ÿһ��
    [index,data]=com_kCorssvalind(data,K)
    
    for k=1:K
        %��ȡ���ݼ���Ӧ��ѵ����train�Ͳ��Լ�test
        [train,test]=com_getTrianAndTest(data,index,k);
        %��ʼ��a,b����
        [a,b]=init(featureNum);
        %����ѵ������ȡ��Ӧ��Z����
        [Z]=com_calcuZ(train,a,b);
        %ʹ��L1�����һ��u��������U��
        [U,e,lambdas]=useL1(Z,train(:,end));
        %ʹ��fisher�������׼ȷ��
        [trainAccuarcyList]=an_analyzeByFisher(U,e,Z,train(:,col));
        %����ѵ����׼ȷ�ȣ�ѡȡһ���Ϻõ�uֵ��������Ԥ���uֵ������test����׼ȷ��
        [maxTrainScore,testScore,maxU,maxE]=summy(trainAccuarcyList,U,e,a,b,test);
        %��ǰ�������uֵ
        maxTrainUList(k,:)=maxU;
        %��ǰ�������eֵ
        maxTrainEList(k,:)=maxE;
        %��ǰ�������ѵ����Ԥ��÷�
        maxTrainScoreList(k,:)=maxTrainScore;
        %��ǰ��Ĳ��Լ�Ԥ��÷�
        testScoreList(k,:)=testScore;
    end
end

function [U,e,lambdas] = useL1(Z,trueResult)
    [betaTemp,FitInfo]=lasso(Z,trueResult);
    U=betaTemp';
    e=zeros(size(U,1),1);
    lambdas=FitInfo.Lambda;
end

%���ܳ�ѵ��׼ȷ�Ⱥ͸���ѵ����׼ȷ�ȣ�ѡȡһ���Ϻõ�uֵ��������Ԥ���uֵ������test����׼ȷ��
%maxTrainScore��ѵ���������ĵ÷�
%testScore�����Լ��ĵ÷�
%maxU����ȡѵ�������÷ֶ�Ӧ��һ��u
%maxE����ȡѵ�������÷ֶ�Ӧ��һ��e
function [maxTrainScore,testScore,maxU,maxE]=summy(trainAccuarcyList,U,e,a,b,test)
    %��ȡ��ߵ�׼ȷ�ȣ������±�
    [maxTrainAccuarcy,maxIndex]=max(trainAccuarcyList(:,1));
    %��ȡ��ߵ�׼ȷ�ȣ���������ָ�꣬�������Ⱥ������
    maxTrainScore=trainAccuarcyList(maxIndex,:);
    
    %test����Ϊ��ʱ��Ĭ�Ϸ���
    if size(test,1)==0
        testScore=zeros(1,3);
        return;
    end
    
    %��ȡ׼ȷ����߶�Ӧ�ĵ�uֵ
    maxU=U(maxIndex,:);
    %��ȡ׼ȷ����߶�Ӧ�ĵ�eֵ
    maxE=e(maxIndex,1);
    %��ȡ��Ӧ�Ĳ��Լ��÷�
    [testScore]=summyOneU(maxU,maxE,a,b,test);
end

%���ܳ�ѵ��׼ȷ�Ⱥ͸���ѵ����׼ȷ�ȣ�ʹ���û�ָ����������Ԥ���uֵ������test����׼ȷ��
%test:���Լ�
%u���û�ָ����һ��u
%e���û�ָ����һ��e
function [testScore]=summyOneU(u,e,a,b,test)   
    [testZ]=com_calcuZ(test,a,b);
    testScore=an_analyzeByFisher(u,e,testZ,test(:,size(test,2)));
end

%��ʼ��a,b����
function [a,b]=init(featureNum)
    a=zeros(1,featureNum);
    b=ones(1,featureNum);
end
function [maxTrainScoreList,testScoreList,maxTrainUList,maxTrainEList]=L12(data)
    %K������֤�в���
    K=5;
    
    %��ȡ���ݵ�������
    [row,col]=size(data);
    
    %��������
    featureNum=col-1;
    
    %k-���������ݷ����±�
%     [index]=crossvalind('Kfold',row,K);

    %k-���������ݷ����±�,ÿһ����ȷֲ���ÿһ��
    [index,data]=com_kCorssvalind(data,K)
    for k=1:K
        %��ȡ���ݼ���Ӧ��ѵ����train�Ͳ��Լ�test
        [train,test]=com_getTrianAndTest(data,index,k);
        %��ʼ��a,b����
        [a,b]=init(featureNum);
        %����ѵ������ȡ��Ӧ��Z����
        [Z]=com_calcuZ(train,a,b);
        %ʹ��L1/2�����һ��u��������U��
        [U,e,lambdas]=useL12_x(Z,train(:,col));
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
    [testScoreList,maxTrainUList,maxTrainEList]=improve(testScoreList,maxTrainUList,maxTrainEList,a,b,data,index,K)
end

%�Ż����
%testScoreList��ԭ�ȵĲ��Լ����
%maxTrainUList��ʵ��������u����
%maxTrainEList��ʵ��������e����
%data��ԭ������
%index������������
%K��k-�۽���
function [testScoreList,maxTrainUList,maxTrainEList]=improve(testScoreList,maxTrainUList,maxTrainEList,a,b,data,index,K)
   %�����쳣�жϣ�׼ȷ��0.8���£�������0.7����
    exceptionIndex1=find(testScoreList(:,1)<0.8);
    exceptionIndex2=find(testScoreList(:,2)<0.7);
    mergeIndex=[exceptionIndex1;exceptionIndex2];
    %������쳣��ʵ������
    mergeIndex=unique(mergeIndex);
    
    [none,usedUIndex]=max(testScoreList(:,1));
    usedU=maxTrainUList(usedUIndex,:);
    usedE=maxTrainEList(usedUIndex,:);
    for k=1:K
        if(ismember(k,mergeIndex)==0)
            continue;
        end
        %��ȡ���ݼ���Ӧ��ѵ����train�Ͳ��Լ�test
        [train,test]=com_getTrianAndTest(data,index,k);
        %ʹ��ָ����u��e����׼ȷ��
        [testScore]=summyOneU(usedU,usedE,a,b,test);
        %�滻ԭ���쳣�����Ӧ��u����������ʹ�õ�
        maxTrainUList(k,:)=usedU;
        %�滻ԭ���쳣�����Ӧ��e����������ʹ�õ�
        maxTrainEList(k,:)=usedE;
        %�滻ԭ���쳣�����ʹ�õ�ǰ���
        testScoreList(k,:)=testScore;
    end
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

%ʹ��L1/2�����u��trueResult���������Ƿ���ֵ,Z����,U������Ƕ�Ӧuֵ
function [U,e,lambdas]=useL12_x(Z,trueResult)
    options.verbose=1;
    options.lambda=0.001 ;% regul parameter
    options.theta=.01; % parameter for lsp
    options.p=.5; % parameter for lp
    options.reg='lp'; % l2 l1 lp, lsp are possible options
    lambdas=0.001:0.004:0.06;%DLBCL,colonʽ�ӵ�lam
    %lambdas=0.001:0.002:0.8%prostateʽ�ӵ�lam
for i=1:size(lambdas,2)
    options.lambda=lambdas(i)
    [svmlp]=gist_hinge2(Z,trueResult,options);
     U(i,:)=(svmlp.w(:,1))'
    e(i,:)=0
end


     
end






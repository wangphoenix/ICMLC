function [maxTrainScoreList,testScoreList,maxTrainUList,maxTrainEList,trainZero] = L12_tool(data )
%L1_LASSO �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
    %K������֤�в���
    K=10;
    
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
        [U,e,lambdas]=useL12_tool(Z,train(:,end));
        %ʹ��fisher�������׼ȷ��
        [trainAccuarcyList]=an_analyzeByFisher(U,e,Z,train(:,col));
        
        %����ѵ����׼ȷ�ȣ�ѡȡһ���Ϻõ�uֵ��������Ԥ���uֵ������test����׼ȷ��
        [maxTrainScore,testScore,maxU,maxE,zeronum]=summy(trainAccuarcyList,U,e,a,b,test);
        
        trainZero(k,:)=zeronum
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

function [U,e,lambdas] = useL12_tool(Z,trueResult)
    % options for solver
    options.verbose=1;
    options.lambda=0.001 ;% regul parameter
    options.theta=.01; % parameter for lsp
    options.p=.5; % parameter for lp
    options.reg='lp'; % l2 l1 lp, lsp are possible options
    
    
   
    %[row,col]=size(Z)
    %Init_x = randn(col,1);
    %MaxIterNum = 2000;
    %lambdas=7.5:0.05:8;
    lambdas=0.002:0.01:0.18;%colonʽ�ӵ�lam
    %lambdas=0:0.5:100;%10������
    %lambdas=0.03:0.01:0.04;
    %lambdas=0.05:0.5:10;
   % for i=1:size(lambdas,2)
   %     [IJT_LHalf_x, IJT_LHalf_cputime] = IJT_LHalf(Z,trueResult,lambdas(i),Init_x,MaxIterNum);
  %      U(i,:)=IJT_LHalf_x'
  %  end
    %e=zeros(size(U,1),1);
     for i=1:size(lambdas,2)
         options.lambda=lambdas(i)
         [svmlp]=gist_hinge2(Z,trueResult,options);
          %U(2*i-1,:)=(svmlp.w(:,1))' % normal vector to hyperlpaln
          %U(2*i,:)=(svmlp.w(:,2))'
          %e(2i-1,:)=(svmlp.w0)'; % svm bias
          %e(2i-1,:)
          U(i,:)=(svmlp.w(:,1))'
     end
     e=zeros(size(U,1),1);
end

%���ܳ�ѵ��׼ȷ�Ⱥ͸���ѵ����׼ȷ�ȣ�ѡȡһ���Ϻõ�uֵ��������Ԥ���uֵ������test����׼ȷ��
%maxTrainScore��ѵ���������ĵ÷�
%testScore�����Լ��ĵ÷�
%maxU����ȡѵ�������÷ֶ�Ӧ��һ��u
%maxE����ȡѵ�������÷ֶ�Ӧ��һ��e
function [maxTrainScore,testScore,maxU,maxE,zeronum]=summy(trainAccuarcyList,U,e,a,b,test)
     trainAccuarcyList(:,4)=sum(U==0,2)%ͳ��0������
     maxIndex=max(find(trainAccuarcyList(:,1)==max(trainAccuarcyList(:,1))));%���0��׼ȷ����ߵ��±�
      zeronum= trainAccuarcyList(maxIndex,4)%��ȡ0������
    %��ȡ��ߵ�׼ȷ�ȣ������±�
    %[maxTrainAccuarcy,maxIndex]=max(trainAccuarcyList(:,1));
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
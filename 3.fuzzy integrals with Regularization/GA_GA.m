function [accuarcyList,UList]=GA_GA(data)
    %K������֤�в���
    K=10;
    
    %��ȡ���ݵ�������
    [row,col]=size(data);
    
    %��������
    featureNum=col-1;
    
    %k-���������ݷ����±�
%   index=crossvalind('Kfold',row,K);

    %k-���������ݷ����±�,ÿһ����ȷֲ���ÿһ��
    [index,data]=com_kCorssvalind(data,K)
    
    for k=1:K
        %��ȡ���ݼ���Ӧ��ѵ����train�Ͳ��Լ�test
        [train,test]=com_getTrianAndTest(data,index,k);
        %��ʼ��a,b����
        [a,b]=init(featureNum);
        %����ѵ������ȡ��Ӧ��Z����
        [Z]=com_calcuZ(train,a,b);
        [U,e]=useGA(Z,train(:,col),featureNum);
        %ʹ��fisher�������׼ȷ��
        [score]=an_analyzeByFisher(U,e,Z,train(:,col));
        accuarcyList(k,:)=score;
        UList(k,:)=U;
    end
end

%ʹ��GA
%Z:Z����
%trueResult��������ʵ����ֵ
%featureNum����������
function [U,e]=useGA(Z,trueResult,featureNum)
    options = gaoptimset('PopulationSize',50,'StallTimeLimit',inf,'StallGenLimit',inf,'Generations',100);
     fitfun=@(U)fitfun(U,0,Z,trueResult)
     index=2^featureNum-1;
     uNum=size(Z,2);
     minList=zeros(1,uNum);
     maxList=ones(1,uNum);
     %�Ľ��ط�����uֵ������0-1֮��
     [U,fval]=ga(fitfun,index,[],[],[],[],minList,maxList,[],options);
     e=0;
end

%��GA��Ϊfitness������ʹ��fisher�������׼ȷ�ȣ�ֻ����׼ȷ�ȱ�����
%ע��GA������Сֵ�ģ������������ֵ����fitness������һ������
function [accuarcy]=fitfun(U,e,Z,trueResult)
    for i=1:size(U,1)
        %��ǰbetaֵ��Ϊ�˺������㣬����tempBeta,ʹ�ú�X��ά����ͬ***
        tempBeta=U(i,:);
        tempBeta=repmat(tempBeta,size(Z,1),1)
        %��ǰbeta0ֵ***
        tempBeta0=e;
        %ʹ�õ�ǰ��beta��beta0������***
        tempResult=dot(tempBeta,Z,2);
        tempResult2=tempResult;
        tempResult2(:,2)=trueResult;
        %ע��GA������Сֵ�ģ������������ֵ����fitness������һ������
        fitnessResult=-1*an_fitness(tempResult2,2);
    end
%     accuarcy=fitnessResult(1,1);
%ƽ��׼ȷ�ȣ�������
    %accuarcy=0.6*fitnessResult(1)+0.4*fitnessResult(2)
    accuarcy=0.1*fitnessResult(1)+0.9*fitnessResult(2)

end


%��ʼ��a,b����
function [a,b]=init(featureNum)
    a=zeros(1,featureNum);
    b=ones(1,featureNum);
end

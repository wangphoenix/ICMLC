function [trainScoreList,testScoreList,maxTrainUList,maxTrainEList] = ANT_ANT2(data)
    %K������֤�в���
    K=10;
    
    %��ȡ���ݵ�������
    [row,col]=size(data);
    
    %��������
    featureNum=col-1;
    
    %k-���������ݷ����±�,ÿһ����ȷֲ���ÿһ��
    [index,data]=com_kCorssvalind(data,K)

    %��������������Ӧ��һ��z��������ZALL�����У����ٺ����ظ�����
    [a,b]=init(featureNum);
    [ZData]=com_calcuZ(data,a,b);
    
    for k=1:K
        %��ȡ���ݼ���Ӧ��ѵ����train�Ͳ��Լ�test
        [train,test]=com_getTrianAndTest(data,index,k);
        
        %��ʼ��a,b����[a,b]=init(featureNum);����Ͳ�ʹ���ˣ���Ϊ�����ظ�����
        
        %��ȡtrainѵ������Ӧ��Z�����test���Լ���Ӧ��Z����
        [ZTrain,ZTest]=com_getTrianAndTest(ZData,index,k);  
        %uΪ��ǰѵ�����ܹ��õ���һ������u,����u���԰�ѵ������׼ȷ�����
        [u,e]=useAnt(ZTrain,train(:,col),featureNum);
        %ʹ��fisher�������ѵ��������ۺϵ÷֣���׼ȷ�ȣ������Ⱥ�����Ⱦ���
        [trainScore]=an_analyzeByFisher(u,e,ZTrain,train(:,col));
        %���ܳ�ѵ��׼ȷ�Ⱥ͸���ѵ����׼ȷ�ȣ�ʹ���û�ָ����������Ԥ���uֵ������test����׼ȷ��
        [testScore]=summyOneU(u,e,a,b,ZTest,test);
        
        
        %��ǰ�������uֵ
        maxTrainUList(k,:)=u;
        %��ǰ�������eֵ
        maxTrainEList(k,:)=e;
        %��ǰ��Ĳ��Լ�Ԥ��÷�
        testScoreList(k,:)=testScore;
        %��ǰ�������ѵ����Ԥ��÷�
        trainScoreList(k,:)=trainScore;
    end
end

%���ܳ�ѵ��׼ȷ�Ⱥ͸���ѵ����׼ȷ�ȣ�ʹ���û�ָ����������Ԥ���uֵ������test����׼ȷ��
%test:���Լ�
%u���û�ָ����һ��u
%e���û�ָ����һ��e
function [testScore]=summyOneU(u,e,a,b,ZTest,test)   
    testScore=an_analyzeByFisher(u,e,ZTest,test(:,size(test,2)));
end





%ʹ����Ⱥ
%Z:Z����
%trueResult��������ʵ����ֵ
%featureNum����������
function [maxAnt,e]=useAnt(Z,trueResult,featureNum)
    m=20;                      %���ϸ���
    T=5;                      %����������
    Rho=0.4;                   %��Ϣ������ϵ��
    P0=0.2;                    %ת�Ƹ��ʳ���
    uMAX=1;                    %��������x�����ֵ
    uMIN=0;                    %��������x����Сֵ
    step=0.1;                  %�ֲ���������
    e=0;
    %%%%%%%%%%%%%%����������ϵĳ�ʼλ��%%%%%%%%%%%%%%
    %��iֻ���ϣ���j��ά������
    for i=1:m
        for j=1:size(Z,2)
            ant(i,j)=(uMIN+(uMAX-uMIN)*rand);
        end
        %ʹ�õ�iֻ���ϵĽ��ȡ��Ӧֵ
        antFit(i,1)=fitfun(ant(i,:),e,Z,trueResult);
        %��Ϣ�ص�����ʼ��ʱ������Ӧֵ����
        Tau(i,1)=exp(antFit(i,1));
    end
    
    %����T�Σ�tΪ��ǰ�ڼ��ε��������˱�ʾ�ڼ��ε����⣬����ʾǰ�벿�ּ���p����ʹ�õ���
    %Tau(i,t),��벿�ִ������Tau��antFit���Ǵ洢��t+1��
    for t=1:T
        %��Ӣ���ϵ��±�
        [value,maxAntIndex]=max(antFit(:,t));
        lamda=1/t;
        [Tau_best,BestIndex]=max(Tau(:,t));
        %%%%%%%%%%%%%%����״̬ת�Ƹ���%%%%%%%%%%%%%%
        for i=1:m
            P(i,t)=(Tau(BestIndex,t)-Tau(i,t))/Tau(BestIndex,t);
        end
        %%%%%%%%%%%%%%λ�ø���%%%%%%%%%%%%%%
        for i=1:m
            %%%%%%%%%%%%%%��Ӣ���Ͼֲ�����%%%%%%%%%%%%%%
            if i==maxAntIndex
                for j=1:size(Z,2)
                    temp(j)=ant(i,j)+(2*rand-1)*step*lamda;
                end
            else
                %%%%%%%%%%%%%%һ������ȫ������%%%%%%%%%%%%%%
                for j=1:size(Z,2)
                    temp(j)=ant(i,j)+(uMAX-uMIN)*lamda;
                end
            end
            %%%%%%%%%%%%%%�߽紦��%%%%%%%%%%%%%%
            for j=1:size(Z,2)
                if temp(j)<uMIN
                    temp(j)=uMIN;
                end
                if temp(j)<uMAX
                    temp(j)=uMAX;
                end
            end
            %%%%%%%%%%%%%%�ж������Ƿ��ƶ�%%%%%%%%%%%%%%
            %tempFit=fitfun(temp,e,Z,trueResult)�µ���Ӧֵ
            %antFit(i,t)�ɵ���Ӧֵ
            %���˧�º����Ӧ��tempFit����ǰ����Ӧ��TauFit(i,t)�ã�����棬�����´�ֱ������ǰ��
            tempFit=fitfun(temp,e,Z,trueResult);
            if tempFit> antFit(i,t)
                ant(i,:)=temp;
                antFit(i,t+1)=tempFit;
            else
                antFit(i,t+1)=antFit(i,t);
            end
        end
        
        %%%%%%%%%%%%%%������Ϣ��%%%%%%%%%%%%%%
        %antFit(i,t+1)�ոո��µ���Ӧֵ
        %antFit(i,t)�ɵ���Ӧֵ
        for i=1:m
            Tau(i,t+1)=(1-Rho)*Tau(i,t)+exp(antFit(i,t+1)); 
        end
        avg(1,t+1)=mean(Tau(:,t+1));
    end%�������
    %ȡ��t�ε��������Tau(:,T+1)�����У���Ϣ����ߵ��±�
    [max_value,max_index]=max(Tau(:,T+1));
    %��Ϣ����߶�Ӧ������,������϶�Ӧ����̽���������Ž�
    maxAnt=ant(max_index,:);           %�൱�����ű���u
    maxScore=antFit(max_index,T+1); %�൱������ֵ�ۺϵ÷֣�����÷��ۺ�׼ȷ�Ⱥ������ȣ�����һ��׼ȷ�Ⱦ�����
%     maxValue=fitfun(u,e,Z,trueResult); %����ֵ
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
        fitnessResult=an_fitness(tempResult2,2);
    end
    %ƽ��׼ȷ�ȣ�������
    accuarcy=0.6*fitnessResult(1)+0.4*fitnessResult(2)
end

%��ʼ��a,b����
function [a,b]=init(featureNum)
    a=zeros(1,featureNum);
    b=ones(1,featureNum);
end
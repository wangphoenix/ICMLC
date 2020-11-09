function [trainScoreList,testScoreList,maxTrainUList,maxTrainEList,maxTrainAList,maxTrainBList] = ANT_ANT(data)
    %K交叉验证中参数
    K=10;
    
    %获取数据的行列数
    [row,col]=size(data);
    
    %特征个数
    featureNum=col-1;
    
    %k-交叉中数据分组下标,每一类均匀分布在每一组
    [index,data]=com_kCorssvalind(data,K)
    
    for k=1:K
        %获取数据集对应的训练集train和测试集test
        [train,test]=com_getTrianAndTest(data,index,k);       
        %u为当前训练集能够得到的一组最优u,这组u可以把训练集的准确度最大化
        [u,e,a,b]=useAnt(train,train(:,col),featureNum);
        
        [ZTrain]=com_calcuZ(train,a,b);
        [ZTest]=com_calcuZ(test,a,b);
        %使用fisher分类分析训练集结果综合得分，有准确度，灵敏度和特异度决定
        [trainScore]=an_analyzeByFisher(u,e,ZTrain,train(:,col));
        %汇总出训练准确度和根据训练的准确度，使用用户指定，来计算预测的u值作用在test集的准确度
        [testScore]=summyOneU(u,e,a,b,ZTest,test);
        
        
        %当前组的最优u值
        maxTrainUList(k,:)=u;
        %当前组的最优u值
        maxTrainAList(k,:)=a;
        %当前组的最优u值
        maxTrainBList(k,:)=b;
        %当前组的最优e值
        maxTrainEList(k,:)=e;
        %当前组的测试集预测得分
        testScoreList(k,:)=testScore;
        %当前组的最优训练集预测得分
        trainScoreList(k,:)=trainScore;
    end
end

%汇总出训练准确度和根据训练的准确度，使用用户指定，来计算预测的u值作用在test集的准确度
%test:测试集
%u：用户指定的一组u
%e：用户指定的一个e
function [testScore]=summyOneU(u,e,a,b,ZTest,test)   
    testScore=an_analyzeByFisher(u,e,ZTest,test(:,size(test,2)));
end





%使用蚁群
%Z:Z矩阵
%trueResult：样本真实分类值
%featureNum：特征数量
function [maxAnt,e,maxA,maxB]=useAnt(train,trueResult,featureNum)
    m=12;                      %蚂蚁个数
    T=5;                     %最大迭代次数
    Rho=0.9;                   %信息素蒸发系数
    P0=0.2;                    %转移概率常熟
    
    uNum=2^featureNum-1;
    uMAX=1;                    %搜索变量x的最大值
    uMIN=0;                    %搜索变量x的最小值
    aMIN=0;                    %搜索变量a的最小值
    bMAX=1;                    %搜索变量b的最大值
    bMIN=-1;                   %搜索变量b的最小值
    
    step=0.1;                  %局部搜索步长
    e=0;
    %%%%%%%%%%%%%%随机设置蚂蚁的初始位置%%%%%%%%%%%%%%
    %第i只蚂蚁，第j个维度属性
    for i=1:m
        for j=1:uNum
            ant(i,j)=(uMIN+(uMAX-uMIN)*rand);
        end
        for j=1:featureNum
            antA(i,j)=aMIN+rand;
            antB(i,j)=(bMIN+(bMAX-bMIN)*rand);
        end
        %使用第i只蚂蚁的解获取适应值
        antFit(i,1)=fitfun(ant(i,:),e,antA(i,:),antB(i,:),train,trueResult);
        %信息素的量初始化时候由适应值决定
        Tau(i,1)=antFit(i,1);
    end
    
    %迭代T次，t为当前第几次迭代。除了表示第几次迭代外，还表示前半部分计算p概率使用的是
    %Tau(i,t),后半部分代码更新Tau和antFit则是存储在t+1中
    for t=1:T
        %保存变量，避免过多计算
        %tempFit=fitfun(U,e,Z,trueResult);
        lamda=1/t;
        [Tau_best,BestIndex]=max(Tau(:,t));
        %%%%%%%%%%%%%%计算状态转移概率%%%%%%%%%%%%%%
        for i=1:m
            P(i,t)=(Tau(BestIndex,t)-Tau(i,t))/Tau(BestIndex,t);
        end
        %%%%%%%%%%%%%%位置更新%%%%%%%%%%%%%%
        for i=1:m
            %%%%%%%%%%%%%%局部搜索%%%%%%%%%%%%%%
            if P(i,t)<P0
                for j=1:uNum
                    temp(j)=ant(i,j)+(2*rand-1)*step*lamda;
                end
                for j=1:featureNum
                    tempA(j)=antA(i,j)+(2*rand-1)*step*lamda;
                    tempB(j)=antB(i,j)+(2*rand-1)*step*lamda;
                end
            else
                %%%%%%%%%%%%%%全局搜索%%%%%%%%%%%%%%
                for j=1:uNum
                    temp(j)=ant(i,j)+(uMAX-uMIN)*(rand-0.5);
                end
                for j=1:featureNum
                    tempA(j)=antA(i,j)+(2*rand-1)*step*lamda;
                    tempB(j)=antB(i,j)+(2*rand-1)*step*lamda;
                end
            end
            %%%%%%%%%%%%%%边界处理%%%%%%%%%%%%%%
            for j=1:uNum
                if temp(j)<uMIN
                    temp(j)=uMIN;
                end
                if temp(j)>uMAX
                    temp(j)=uMAX;
                end
            end
            for j=1:featureNum
                if tempA(j)<uMIN
                   tempA(j)=uMIN;
                end
                if tempB(j)<bMIN
                   tempB(j)=bMIN;
                end
                if tempB(j)>bMIN
                   tempB(j)=bMIN;
                end
            end
            %%%%%%%%%%%%%%判断蚂蚁是否移动%%%%%%%%%%%%%%
            %tempFit=fitfun(temp,e,Z,trueResult)新的适应值
            %antFit(i,t)旧的适应值
            %如果帅新后的适应度tempFit比以前的适应度TauFit(i,t)好，则代替，否则下次直接用以前的
            tempFit=fitfun(temp,e,tempA,tempB,train,trueResult);
            if tempFit> antFit(i,t)
                ant(i,:)=temp;
                antA(i,:)=tempA;
                antB(i,:)=tempB;
                antFit(i,t+1)=tempFit;
            else
                antFit(i,t+1)=antFit(i,t);
            end
        end
        
        %%%%%%%%%%%%%%更新信息素%%%%%%%%%%%%%%
        %antFit(i,t+1)刚刚更新的适应值
        %antFit(i,t)旧的适应值
        for i=1:m
            Tau(i,t+1)=(1-Rho)*Tau(i,t)+antFit(i,t+1); 
        end
        avg(1,t+1)=mean(Tau(:,t+1));
    end%迭代完成
    %取第t次迭代求出的Tau(:,T+1)数组中，信息素最高的下标
    [max_value,max_index]=max(Tau(:,T+1));
    %信息素最高对应的蚂蚁,这个蚂蚁对应的是探索到的最优解
    maxAnt=ant(max_index,:);           %相当于最优变量u
    maxA=antA(max_index,:);
    maxB=antB(max_index,:);
    maxScore=antFit(max_index,T+1); %相当于最优值综合得分，这个得分综合准确度和灵敏度，不是一个准确度决定的
%     maxValue=fitfun(u,e,Z,trueResult); %最优值
end



%给GA作为fitness函数，使用fisher分类分析准确度，只返回准确度标量，
%注意GA是求最小值的，这里变成求最大值就在fitness函数加一个负号
function [accuarcy]=fitfun(U,e,a,b,train,trueResult)
    [Z]=com_calcuZ(train,a,b);
    %当前beta值，为了后面运算，扩增tempBeta,使得和X的维度相同***
    tempBeta=U;
    tempBeta=repmat(tempBeta,size(Z,1),1)
    %当前beta0值***
    tempBeta0=e;
    %现场求出Z矩阵和当前的beta和beta0求出结果
    
    tempResult=dot(tempBeta,Z,2);
    tempResult2=tempResult;
    tempResult2(:,2)=trueResult;
    fitnessResult=an_fitness(tempResult2,2);
    %平衡准确度，灵敏度
    accuarcy=0.6*fitnessResult(1)+0.4*fitnessResult(2)
end
function [accuarcyList,UList]=GA_GA(data)
    %K交叉验证中参数
    K=10;
    
    %获取数据的行列数
    [row,col]=size(data);
    
    %特征个数
    featureNum=col-1;
    
    %k-交叉中数据分组下标
%   index=crossvalind('Kfold',row,K);

    %k-交叉中数据分组下标,每一类均匀分布在每一组
    [index,data]=com_kCorssvalind(data,K)
    
    for k=1:K
        %获取数据集对应的训练集train和测试集test
        [train,test]=com_getTrianAndTest(data,index,k);
        %初始化a,b参数
        [a,b]=init(featureNum);
        %根据训练集获取对应的Z矩阵
        [Z]=com_calcuZ(train,a,b);
        [U,e]=useGA(Z,train(:,col),featureNum);
        %使用fisher分类分析准确度
        [score]=an_analyzeByFisher(U,e,Z,train(:,col));
        accuarcyList(k,:)=score;
        UList(k,:)=U;
    end
end

%使用GA
%Z:Z矩阵
%trueResult：样本真实分类值
%featureNum：特征数量
function [U,e]=useGA(Z,trueResult,featureNum)
    options = gaoptimset('PopulationSize',50,'StallTimeLimit',inf,'StallGenLimit',inf,'Generations',100);
     fitfun=@(U)fitfun(U,0,Z,trueResult)
     index=2^featureNum-1;
     uNum=size(Z,2);
     minList=zeros(1,uNum);
     maxList=ones(1,uNum);
     %改进地方，吧u值控制在0-1之间
     [U,fval]=ga(fitfun,index,[],[],[],[],minList,maxList,[],options);
     e=0;
end

%给GA作为fitness函数，使用fisher分类分析准确度，只返回准确度标量，
%注意GA是求最小值的，这里变成求最大值就在fitness函数加一个负号
function [accuarcy]=fitfun(U,e,Z,trueResult)
    for i=1:size(U,1)
        %当前beta值，为了后面运算，扩增tempBeta,使得和X的维度相同***
        tempBeta=U(i,:);
        tempBeta=repmat(tempBeta,size(Z,1),1)
        %当前beta0值***
        tempBeta0=e;
        %使用当前的beta和beta0求出结果***
        tempResult=dot(tempBeta,Z,2);
        tempResult2=tempResult;
        tempResult2(:,2)=trueResult;
        %注意GA是求最小值的，这里变成求最大值就在fitness函数加一个负号
        fitnessResult=-1*an_fitness(tempResult2,2);
    end
%     accuarcy=fitnessResult(1,1);
%平衡准确度，灵敏度
    %accuarcy=0.6*fitnessResult(1)+0.4*fitnessResult(2)
    accuarcy=0.1*fitnessResult(1)+0.9*fitnessResult(2)

end


%初始化a,b参数
function [a,b]=init(featureNum)
    a=zeros(1,featureNum);
    b=ones(1,featureNum);
end

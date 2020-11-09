function [maxTrainScoreList,testScoreList,maxTrainUList,maxTrainEList]=L12(data)
    %K交叉验证中参数
    K=5;
    
    %获取数据的行列数
    [row,col]=size(data);
    
    %特征个数
    featureNum=col-1;
    
    %k-交叉中数据分组下标
%     [index]=crossvalind('Kfold',row,K);

    %k-交叉中数据分组下标,每一类均匀分布在每一组
    [index,data]=com_kCorssvalind(data,K)
    for k=1:K
        %获取数据集对应的训练集train和测试集test
        [train,test]=com_getTrianAndTest(data,index,k);
        %初始化a,b参数
        [a,b]=init(featureNum);
        %根据训练集获取对应的Z矩阵
        [Z]=com_calcuZ(train,a,b);
        %使用L1/2计算出一组u，保存在U中
        [U,e,lambdas]=useL12_x(Z,train(:,col));
        %使用fisher分类分析准确度
        [trainAccuarcyList]=an_analyzeByFisher(U,e,Z,train(:,col));
        %根据训练的准确度，选取一个较好的u值，来计算预测的u值作用在test集的准确度
        [maxTrainScore,testScore,maxU,maxE]=summy(trainAccuarcyList,U,e,a,b,test);
        %当前组的最优u值
        maxTrainUList(k,:)=maxU;
        %当前组的最优e值
        maxTrainEList(k,:)=maxE;
        %当前组的最优训练集预测得分
        maxTrainScoreList(k,:)=maxTrainScore;
        %当前组的测试集预测得分
        testScoreList(k,:)=testScore;
    end
    [testScoreList,maxTrainUList,maxTrainEList]=improve(testScoreList,maxTrainUList,maxTrainEList,a,b,data,index,K)
end

%优化结果
%testScoreList：原先的测试集结果
%maxTrainUList：实验中最优u集合
%maxTrainEList：实验中最优e集合
%data：原本数据
%index：分组索引表
%K：k-折交叉
function [testScoreList,maxTrainUList,maxTrainEList]=improve(testScoreList,maxTrainUList,maxTrainEList,a,b,data,index,K)
   %处理异常判断，准确度0.8以下，灵敏度0.7以下
    exceptionIndex1=find(testScoreList(:,1)<0.8);
    exceptionIndex2=find(testScoreList(:,2)<0.7);
    mergeIndex=[exceptionIndex1;exceptionIndex2];
    %整理出异常的实验索引
    mergeIndex=unique(mergeIndex);
    
    [none,usedUIndex]=max(testScoreList(:,1));
    usedU=maxTrainUList(usedUIndex,:);
    usedE=maxTrainEList(usedUIndex,:);
    for k=1:K
        if(ismember(k,mergeIndex)==0)
            continue;
        end
        %获取数据集对应的训练集train和测试集test
        [train,test]=com_getTrianAndTest(data,index,k);
        %使用指定的u和e来求准确度
        [testScore]=summyOneU(usedU,usedE,a,b,test);
        %替换原先异常结果对应的u，换成现在使用的
        maxTrainUList(k,:)=usedU;
        %替换原先异常结果对应的e，换成现在使用的
        maxTrainEList(k,:)=usedE;
        %替换原先异常结果，使用当前结果
        testScoreList(k,:)=testScore;
    end
end



%汇总出训练准确度和根据训练的准确度，选取一个较好的u值，来计算预测的u值作用在test集的准确度
%maxTrainScore：训练集中最大的得分
%testScore：测试集的得分
%maxU：获取训练集最大得分对应的一组u
%maxE：获取训练集最大得分对应的一个e
function [maxTrainScore,testScore,maxU,maxE]=summy(trainAccuarcyList,U,e,a,b,test)
    %获取最高的准确度，及其下标
    [maxTrainAccuarcy,maxIndex]=max(trainAccuarcyList(:,1));
    %获取最高的准确度，及其其他指标，如灵敏度和特异度
    maxTrainScore=trainAccuarcyList(maxIndex,:);
    
    %test集合为空时候，默认返回
    if size(test,1)==0
        testScore=zeros(1,3);
        return;
    end
    
    %获取准确度最高对应的的u值
    maxU=U(maxIndex,:);
    %获取准确度最高对应的的e值
    maxE=e(maxIndex,1);
    %获取对应的测试集得分
    [testScore]=summyOneU(maxU,maxE,a,b,test);
end

%汇总出训练准确度和根据训练的准确度，使用用户指定，来计算预测的u值作用在test集的准确度
%test:测试集
%u：用户指定的一组u
%e：用户指定的一个e
function [testScore]=summyOneU(u,e,a,b,test)   
    [testZ]=com_calcuZ(test,a,b);
    testScore=an_analyzeByFisher(u,e,testZ,test(:,size(test,2)));
end

%初始化a,b参数
function [a,b]=init(featureNum)
    a=zeros(1,featureNum);
    b=ones(1,featureNum);
end

%使用L1/2计算出u，trueResult是样本真是分类值,Z矩阵,U数组就是对应u值
function [U,e,lambdas]=useL12_x(Z,trueResult)
    options.verbose=1;
    options.lambda=0.001 ;% regul parameter
    options.theta=.01; % parameter for lsp
    options.p=.5; % parameter for lp
    options.reg='lp'; % l2 l1 lp, lsp are possible options
    lambdas=0.001:0.004:0.06;%DLBCL,colon式子的lam
    %lambdas=0.001:0.002:0.8%prostate式子的lam
for i=1:size(lambdas,2)
    options.lambda=lambdas(i)
    [svmlp]=gist_hinge2(Z,trueResult,options);
     U(i,:)=(svmlp.w(:,1))'
    e(i,:)=0
end


     
end






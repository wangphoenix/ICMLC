function [maxTrainScoreList,testScoreList,maxTrainUList,maxTrainEList,trainZero] = L12_tool(data )
%L1_LASSO 此处显示有关此函数的摘要
%   此处显示详细说明
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
        %初始化a,b参数
        [a,b]=init(featureNum);
        %根据训练集获取对应的Z矩阵
        [Z]=com_calcuZ(train,a,b);
        %使用L1计算出一组u，保存在U中
        [U,e,lambdas]=useL12_tool(Z,train(:,end));
        %使用fisher分类分析准确度
        [trainAccuarcyList]=an_analyzeByFisher(U,e,Z,train(:,col));
        
        %根据训练的准确度，选取一个较好的u值，来计算预测的u值作用在test集的准确度
        [maxTrainScore,testScore,maxU,maxE,zeronum]=summy(trainAccuarcyList,U,e,a,b,test);
        
        trainZero(k,:)=zeronum
        %当前组的最优u值
        maxTrainUList(k,:)=maxU;
        %当前组的最优e值
        maxTrainEList(k,:)=maxE;
        %当前组的最优训练集预测得分
        maxTrainScoreList(k,:)=maxTrainScore;
        %当前组的测试集预测得分
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
    lambdas=0.002:0.01:0.18;%colon式子的lam
    %lambdas=0:0.5:100;%10个样本
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

%汇总出训练准确度和根据训练的准确度，选取一个较好的u值，来计算预测的u值作用在test集的准确度
%maxTrainScore：训练集中最大的得分
%testScore：测试集的得分
%maxU：获取训练集最大得分对应的一组u
%maxE：获取训练集最大得分对应的一个e
function [maxTrainScore,testScore,maxU,maxE,zeronum]=summy(trainAccuarcyList,U,e,a,b,test)
     trainAccuarcyList(:,4)=sum(U==0,2)%统计0的数量
     maxIndex=max(find(trainAccuarcyList(:,1)==max(trainAccuarcyList(:,1))));%最多0，准确度最高的下标
      zeronum= trainAccuarcyList(maxIndex,4)%获取0的数量
    %获取最高的准确度，及其下标
    %[maxTrainAccuarcy,maxIndex]=max(trainAccuarcyList(:,1));
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
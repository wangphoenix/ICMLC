clear all;

data=load('data23.txt');
[data]=main_prepare(data)
c=[]
K=3;
for k=1:K
    [maxTrainScoreList,testScoreList,maxTrainUList,maxTrainEList,trainZero]=L12_tool(data)
    avgAcc=mean(testScoreList);
    avgZero=mean(trainZero);
    a(k,:)=avgAcc;
    b(k,:)=avgZero;
    c=[c;maxTrainUList]
end

avgA=mean(a)
avgZero=mean(b)

%[maxTrainScoreList,testScoreList,maxTrainUList,maxTrainEList]=L2_xian(data)
%sum(c)求每一列的和
%[maxU,index]=max(sum(c))%求每一列中和最大以及最大值对应的下标



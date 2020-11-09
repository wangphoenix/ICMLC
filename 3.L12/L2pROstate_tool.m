clear all;

data=load('data4.txt');
[data]=main_prepare(data)
c=[]
K=20;
for k=1:K
    [maxTrainScoreList,testScoreList,maxTrainUList,maxTrainEList,trainZero]=L2_tool(data)
    avgAcc=mean(testScoreList);
    avgZero=mean(trainZero);
    a(k,:)=avgAcc;
    b(k,:)=avgZero;
    c=[c;maxTrainUList]
end

avgA=mean(a)
avgZero=mean(b)

%[maxTrainScoreList,testScoreList,maxTrainUList,maxTrainEList]=L2_xian(data)



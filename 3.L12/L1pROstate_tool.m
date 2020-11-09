clear all;

data=load('data2.txt');
[data]=main_prepare(data)

%西安的学生
%[maxTrainScoreList,testScoreList,maxTrainUList,maxTrainEList]=L2_xian(data)
c=[]
K=20;
for k=1:K
    [maxTrainScoreList,testScoreList,maxTrainUList,maxTrainEList,trainZero]=L1_tool(data)
    avgAcc=mean(testScoreList);
    avgZero=mean(trainZero);
    a(k,:)=avgAcc;
    b(k,:)=avgZero;
    c=[c;maxTrainUList]
end

avgA=mean(a)
avgZero=mean(b)
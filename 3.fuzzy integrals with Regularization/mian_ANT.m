%使用GA求解

%清空所有变量
clear all;
%导入数据集
data=load('L12+L2-3.txt');
[data]=main_prepare(data)
[trainScoreList,testScoreList,maxTrainUList,maxTrainEList]=ANT_ANT1(data)
avg=mean(testScoreList);
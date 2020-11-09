%使用GA求解

%清空所有变量
clear all;
%导入数据集
data=load('data7.txt');
[data]=main_prepare(data)
[trainScoreList,testScoreList,maxTrainUList,maxTrainEList,maxTrainAList,maxTrainBList]=ANT_ANT3(data)
avg=mean(testScoreList);
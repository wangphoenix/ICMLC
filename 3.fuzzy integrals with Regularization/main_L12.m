%清空所有变量
clear all;

%导入数据集
%data=load('data1.txt');这个数据集效果最好
%data=load('postateAll.txt');
data=load('data5.txt')
[data]=main_prepare(data)
[maxTrainScoreList,testScoreList,maxTrainUList,maxTrainEList]=L12_L12(data)
avg=mean(testScoreList);
% accuarcyList=GA_GA(data)

 



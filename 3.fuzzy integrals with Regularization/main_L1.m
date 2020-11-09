%使用L1求解

%清空所有变量
clear all;
%导入数据集
data=load('data20.txt');
%data=load('postateAll.txt');
%data=load('data4.txt');
[data]=main_prepare(data)
[maxTrainScoreList,testScoreList,maxTrainUList,maxTrainEList]=L1_Lasso(data)
avg=mean(testScoreList);
% accuarcyList=GA_GA(data)

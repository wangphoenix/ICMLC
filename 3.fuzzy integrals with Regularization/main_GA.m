%使用GA求解

%清空所有变量
clear all;
%导入数据集
%data=load('data1.txt');
[data]=main_prepare(data)
[accuarcyList,UList]=GA_GA(data)
% accuarcyList=GA_GA(data)

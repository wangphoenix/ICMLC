%������б���
clear all;

%�������ݼ�
%data=load('data1.txt');������ݼ�Ч�����
%data=load('postateAll.txt');
data=load('data5.txt')
[data]=main_prepare(data)
[maxTrainScoreList,testScoreList,maxTrainUList,maxTrainEList]=L12_L12(data)
avg=mean(testScoreList);
% accuarcyList=GA_GA(data)

 



%ʹ��GA���

%������б���
clear all;
%�������ݼ�
data=load('L12+L2-3.txt');
[data]=main_prepare(data)
[trainScoreList,testScoreList,maxTrainUList,maxTrainEList]=ANT_ANT1(data)
avg=mean(testScoreList);
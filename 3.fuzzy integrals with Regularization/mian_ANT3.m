%ʹ��GA���

%������б���
clear all;
%�������ݼ�
data=load('data7.txt');
[data]=main_prepare(data)
[trainScoreList,testScoreList,maxTrainUList,maxTrainEList,maxTrainAList,maxTrainBList]=ANT_ANT3(data)
avg=mean(testScoreList);
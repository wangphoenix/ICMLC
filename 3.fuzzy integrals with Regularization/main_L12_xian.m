%��ҵ�汾

%������б���
clear all;

%�������ݼ�
%data=load('G:\��ҵʵ��ɹ�\����\1.GCRMA����������\2.Prostate\L12-15.txt');
%data=load('G:\��ҵʵ��ɹ�\����\1.GCRMA����������\3.colon\L12-15.txt');
data=load('G:\��ҵʵ��ɹ�\����\1.GCRMA����������\4.GLI-85\L12-15.txt');
[data]=main_prepare(data)
[maxTrainScoreList,testScoreList,maxTrainUList,maxTrainEList]=L12_L12_xian(data)
avg=mean(testScoreList);

%sum(maxTrainUList==0,2)��ȡÿ�ν��0������
%I=find(maxTrainUList(5,:)~=0) ��ȡ��10�в�Ϊ0�������±�
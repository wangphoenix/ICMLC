%毕业版本

%清空所有变量
clear all;

%导入数据集
%data=load('G:\毕业实验成果\数据\1.GCRMA处理后的数据\2.Prostate\L12-15.txt');
%data=load('G:\毕业实验成果\数据\1.GCRMA处理后的数据\3.colon\L12-15.txt');
data=load('G:\毕业实验成果\数据\1.GCRMA处理后的数据\4.GLI-85\L12-15.txt');
[data]=main_prepare(data)
[maxTrainScoreList,testScoreList,maxTrainUList,maxTrainEList]=L12_L12_xian(data)
avg=mean(testScoreList);

%sum(maxTrainUList==0,2)获取每次结果0的数量
%I=find(maxTrainUList(5,:)~=0) 获取第10行不为0的所有下标
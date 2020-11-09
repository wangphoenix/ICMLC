function [mergeIndex,mergeData] = com_kCorssvalind(data,k)
%crossvalind k-折交叉验证分组方法的改写
%由于原始数据样本少，所以使用原始的k-折交叉验证分组会容易引起极端情况，
%也就是说一些组只有某一类，其他类别样本不包含，改写后会平均分布
%data：原始数据
%k：k组
%index：k-交叉中数据分组下标

%从最后一列中，获取数据集中的不重复的分类结果
class=unique(data(:,end));

%AGroup标记了第一类
groupA=ismember(data(:,end),class(1,1))
%BGroup标记了第二类
groupB=ismember(data(:,end),class(2,1))

%第一类的样本
dataA=data(groupA,:);
%第二类的样本
dataB=data(groupB,:);
dataB(:,end)=dataB(:,end)+0;
%合并两类样本
mergeData=[dataA;dataB];
%第一类的样本的k折交叉下标
indexA=crossvalind('Kfold',size(dataA,1),k);
%第二类的样本的k折交叉下标
indexB=crossvalind('Kfold',size(dataB,1),k);
%合并k折交叉下标，使得和mergeData下标统一
mergeIndex=[indexA;indexB];
end


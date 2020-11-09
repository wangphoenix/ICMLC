function [mergeIndex,mergeData] = com_kCorssvalind(data,k)
%crossvalind k-�۽�����֤���鷽���ĸ�д
%����ԭʼ���������٣�����ʹ��ԭʼ��k-�۽�����֤������������𼫶������
%Ҳ����˵һЩ��ֻ��ĳһ�࣬���������������������д���ƽ���ֲ�
%data��ԭʼ����
%k��k��
%index��k-���������ݷ����±�

%�����һ���У���ȡ���ݼ��еĲ��ظ��ķ�����
class=unique(data(:,end));

%AGroup����˵�һ��
groupA=ismember(data(:,end),class(1,1))
%BGroup����˵ڶ���
groupB=ismember(data(:,end),class(2,1))

%��һ�������
dataA=data(groupA,:);
%�ڶ��������
dataB=data(groupB,:);
dataB(:,end)=dataB(:,end)+0;
%�ϲ���������
mergeData=[dataA;dataB];
%��һ���������k�۽����±�
indexA=crossvalind('Kfold',size(dataA,1),k);
%�ڶ����������k�۽����±�
indexB=crossvalind('Kfold',size(dataB,1),k);
%�ϲ�k�۽����±꣬ʹ�ú�mergeData�±�ͳһ
mergeIndex=[indexA;indexB];
end


function [train_data,test_data]=com_getTrianAndTest(data,index,k)
%����kֵ��ȡ���ݼ���Ӧ��ѵ�����ϲ��Լ�
%data:���ݼ�
%index��k-���������ݷ����±�
%k:�ڼ�����Ϊtest����Ҳ���ǲ��Լ�
%train:ѵ����
%test:���Լ�


    %��ȡ�±����k���к�
    test = (index == k)
    %��ȡ�±겻����k���к�
    train = ~test
    %��ȡ�±����k���кŵ����ݣ���Ϊ���Լ�
    test_data=data(test,:)
    %��ȡ�±겻����k���кţ���Ϊѵ����
    train_data=data(train,:)
end
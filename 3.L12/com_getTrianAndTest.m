function [train_data,test_data]=com_getTrianAndTest(data,index,k)
%根据k值获取数据集对应的训练集合测试集
%data:数据集
%index：k-交叉中数据分组下标
%k:第几组作为test集，也就是测试集
%train:训练集
%test:测试集


    %获取下标等于k的行号
    test = (index == k)
    %获取下标不等于k的行号
    train = ~test
    %获取下标等于k的行号的数据，作为测试集
    test_data=data(test,:)
    %获取下标不等于k的行号，作为训练集
    train_data=data(train,:)
end
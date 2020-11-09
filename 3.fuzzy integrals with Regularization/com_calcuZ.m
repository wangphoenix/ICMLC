%获取训练集对应的Z矩阵
function[Z] = com_calcuZ(train,a,b)
    %获取数据的行列数
    [row,col]=size(train);
    %特征个数
    featureNum=col-1;
    
    %Z矩阵中列下标是从0开始，但是matlab小标是从1开始，但是恰好列下标为0的值都为1，所以不用管了
    %j为Z矩阵的行小标
    for j=1:row
        %k为Z矩阵的列下标，Z矩阵中，3特征有7列，4特征15列
        for k=1:2^featureNum-1
            [Z(j,k)]=calcu_z(train,a,b,j,k,featureNum);
        end
    end
end

%根据k值，计算出对应的z值
%train：训练集
%j：train数据集的第j行，或者Z矩阵的第j行
%k:Z矩阵的第k列
%featureNum：特征个数
%z:Z(j,k)对应的值，z值是根据k和train上第j行数据计算出来的
function[z] = calcu_z(train,a,b,j,k,featureNum)
    %k对应的二进制，编码的位数等于样本特征个数
    kBit=dec2bin(k,featureNum);
    %做一个细节处理，这个二进制是要逆序的，因为循环时候k是从001-111，
    %而实际上是编码要从100，010,110,001,101,001,111，刚好是k原来值的逆序
    kBit=fliplr(kBit)
    min=100000.0;
    max=-100000.0;
    %这个标志位是考虑k编码为全1的情况，如果全1，那么就是没有计算过max的值，没有计算过max那默认等于0
    tag=0;
    %表示第几个特征，k值的二进制位，求一个z值，
    %循环得到k值编码中，一个1位上最小的特征值，和0位上最大的特征值
    for i=1:featureNum
        z=a(i)+b(i)*train(j,i);
                
        %根据公式，等于1位，对应的特征值，和最小值比较
        if(kBit(i)=='1')
            if z<min
                min=z;
            end
        %根据公式，等于0位，对应的特征值，和最大值比较
        else
            tag=1;
            if z>max
                max=z;
            end
         end    
    end 
    %如果tag等于0，说明没有计算过max的值，没有计算过max那默认等于0
    if tag==0
        max=0.0;
    end
    %根据公式，如果min-max>0，则z=min-max；否则，z=0
    if (min-max)>0
        z=min-max;
    else
        z=0;
    end
end
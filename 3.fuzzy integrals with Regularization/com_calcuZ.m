%��ȡѵ������Ӧ��Z����
function[Z] = com_calcuZ(train,a,b)
    %��ȡ���ݵ�������
    [row,col]=size(train);
    %��������
    featureNum=col-1;
    
    %Z���������±��Ǵ�0��ʼ������matlabС���Ǵ�1��ʼ������ǡ�����±�Ϊ0��ֵ��Ϊ1�����Բ��ù���
    %jΪZ�������С��
    for j=1:row
        %kΪZ��������±꣬Z�����У�3������7�У�4����15��
        for k=1:2^featureNum-1
            [Z(j,k)]=calcu_z(train,a,b,j,k,featureNum);
        end
    end
end

%����kֵ���������Ӧ��zֵ
%train��ѵ����
%j��train���ݼ��ĵ�j�У�����Z����ĵ�j��
%k:Z����ĵ�k��
%featureNum����������
%z:Z(j,k)��Ӧ��ֵ��zֵ�Ǹ���k��train�ϵ�j�����ݼ��������
function[z] = calcu_z(train,a,b,j,k,featureNum)
    %k��Ӧ�Ķ����ƣ������λ������������������
    kBit=dec2bin(k,featureNum);
    %��һ��ϸ�ڴ��������������Ҫ����ģ���Ϊѭ��ʱ��k�Ǵ�001-111��
    %��ʵ�����Ǳ���Ҫ��100��010,110,001,101,001,111���պ���kԭ��ֵ������
    kBit=fliplr(kBit)
    min=100000.0;
    max=-100000.0;
    %�����־λ�ǿ���k����Ϊȫ1����������ȫ1����ô����û�м����max��ֵ��û�м����max��Ĭ�ϵ���0
    tag=0;
    %��ʾ�ڼ���������kֵ�Ķ�����λ����һ��zֵ��
    %ѭ���õ�kֵ�����У�һ��1λ����С������ֵ����0λ����������ֵ
    for i=1:featureNum
        z=a(i)+b(i)*train(j,i);
                
        %���ݹ�ʽ������1λ����Ӧ������ֵ������Сֵ�Ƚ�
        if(kBit(i)=='1')
            if z<min
                min=z;
            end
        %���ݹ�ʽ������0λ����Ӧ������ֵ�������ֵ�Ƚ�
        else
            tag=1;
            if z>max
                max=z;
            end
         end    
    end 
    %���tag����0��˵��û�м����max��ֵ��û�м����max��Ĭ�ϵ���0
    if tag==0
        max=0.0;
    end
    %���ݹ�ʽ�����min-max>0����z=min-max������z=0
    if (min-max)>0
        z=min-max;
    else
        z=0;
    end
end
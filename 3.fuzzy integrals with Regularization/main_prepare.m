function [ processedData ] = main_prepare( data )
%������Ԥ�������ع���
%data������ǰ������
%processedData������������
   
    %��Ҫϸ�ڣ�������������ֵͳһ��1,1��ʾ������2��ʾ���������ӡ������Ŵ��㷨����
    data(:,end)=data(:,end)+1;
    processedData=data;
end


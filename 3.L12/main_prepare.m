function [ processedData ] = main_prepare( data )
%做数据预处理的相关工作
%data：处理前的数据
%processedData：处理后的数据
   
    %重要细节：这个必须给分类值统一加1,1表示正常，2表示患病这样子。否则遗传算法出错
    data(:,end)=data(:,end)+1;
    processedData=data;
end

